from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

SLOTS_PER_DAY = 96
LOOK_BACK_DAYS = 14

DEFAULT_PATTERNS = (
    "loadshape_*.csv", "pvshape_*.csv", "EVAC_*.csv", "EVDC_*.csv", "V2G_*.csv",
)

@dataclass
class PipelineConfig:
    project_dir: Path = Path(".")
    csv_inputs_dir: Path = Path("csv_inputs")
    work_dir: Path = Path("vpp_forecast_output")
    master_dss: Path = Path("master_with_ev_pv_v2g.dss")
    vpu_module: Path = Path("investigate_Vpu.py")
    random_seed: int = 42
    scenario_days: int = 365
    vpp_threshold_pu: float = 0.95
    patterns: Sequence[str] = DEFAULT_PATTERNS


def read_1col(path: Path) -> np.ndarray:
    vals = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            token = line.split(",", 1)[0].strip()
            if not token:
                continue
            try:
                vals.append(float(token))
            except ValueError:
                continue
    if not vals:
        raise ValueError(f"No numeric first-column data: {path}")
    return np.asarray(vals, dtype=np.float64)


def discover_input_files(input_dir: Path, patterns: Sequence[str]) -> list[Path]:
    unique = {}
    for pattern in patterns:
        for p in input_dir.glob(pattern):
            if p.is_file():
                unique[p.resolve()] = p
    files = sorted(unique.values(), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No matching CSV files under {input_dir.resolve()}")
    return files


def load_aligned_inputs(input_dir: Path, patterns: Sequence[str]):
    files = discover_input_files(input_dir, patterns)
    arrays = [read_1col(p) for p in files]
    lengths = [len(a) for a in arrays]
    min_len = min(lengths)
    min_len -= min_len % SLOTS_PER_DAY
    if min_len < (LOOK_BACK_DAYS + 1) * SLOTS_PER_DAY:
        raise ValueError("Need at least 15 complete days of 15-minute data.")
    if len(set(lengths)) != 1:
        print(f"[WARN] CSV lengths differ {sorted(set(lengths))}; truncating to {min_len}.")
    return files, np.column_stack([a[:min_len] for a in arrays])


def save_total(matrix: np.ndarray, path: Path) -> np.ndarray:
    total = np.sum(matrix, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"total": total}).to_csv(path, index=False, header=False)
    return total


def build_xgb_dataset(values: np.ndarray):
    v = np.asarray(values, dtype=np.float32)
    start = LOOK_BACK_DAYS * SLOTS_PER_DAY
    X, y = [], []
    for i in range(start, len(v)):
        slot = i % SLOTS_PER_DAY
        same_slot_14d = v[i-start:i:SLOTS_PER_DAY]
        recent4 = v[i-4:i]
        X.append(np.r_[same_slot_14d, recent4, slot,
                       np.sin(2*np.pi*slot/SLOTS_PER_DAY),
                       np.cos(2*np.pi*slot/SLOTS_PER_DAY)])
        y.append(v[i])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def train_xgboost(total: np.ndarray, seed: int = 42) -> XGBRegressor:
    X, y = build_xgb_dataset(total)
    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=500, max_depth=5,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X, y)
    return model


def bootstrap_complete_days(matrix: np.ndarray, n_days: int, seed: int):
    source_days = len(matrix) // SLOTS_PER_DAY
    cube = matrix[:source_days*SLOTS_PER_DAY].reshape(source_days, SLOTS_PER_DAY, matrix.shape[1])
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, source_days, size=n_days)
    return cube[chosen].reshape(n_days*SLOTS_PER_DAY, matrix.shape[1]), chosen


def forecast_next_days(model: XGBRegressor, scenario_total: np.ndarray) -> np.ndarray:
    n = len(scenario_total)
    pred = np.full(n, np.nan, dtype=float)
    history = list(np.asarray(scenario_total[:LOOK_BACK_DAYS*SLOTS_PER_DAY], dtype=float))
    n_days = n // SLOTS_PER_DAY
    for day in range(LOOK_BACK_DAYS, n_days):
        day_pred = []
        for slot in range(SLOTS_PER_DAY):
            h = np.asarray(history + day_pred, dtype=np.float32)
            i = len(h)
            same14 = h[i-LOOK_BACK_DAYS*SLOTS_PER_DAY:i:SLOTS_PER_DAY]
            recent4 = h[i-4:i]
            feat = np.r_[same14, recent4, slot,
                         np.sin(2*np.pi*slot/SLOTS_PER_DAY),
                         np.cos(2*np.pi*slot/SLOTS_PER_DAY)].reshape(1, -1)
            day_pred.append(float(model.predict(feat)[0]))
        s = day*SLOTS_PER_DAY
        pred[s:s+SLOTS_PER_DAY] = day_pred
        history.extend(day_pred)
        history = history[-LOOK_BACK_DAYS*SLOTS_PER_DAY:]
    return pred


def redistribute_forecast(scenario_matrix: np.ndarray, forecast_total: np.ndarray) -> np.ndarray:
    out = scenario_matrix.astype(float).copy()
    base = out.sum(axis=1)
    valid = np.isfinite(forecast_total) & (np.abs(base) > 1e-12)
    out[valid] *= (forecast_total[valid] / base[valid])[:, None]
    return out


def write_profiles(files: list[Path], matrix: np.ndarray, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, src in enumerate(files):
        pd.Series(matrix[:, j]).to_csv(out_dir/src.name, index=False, header=False)

def _load_vpu_module(path):
    import importlib.util
    import sys

    module_name = "investigate_Vpu"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Vpu module: {path}")

    mod = importlib.util.module_from_spec(spec)

    # 중요: exec_module 전에 등록
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)
    return mod

def run_opendss_vpp(cfg: PipelineConfig, generated_dir: Path, target_buses: Sequence[str] | None):
    vpu = _load_vpu_module((cfg.project_dir/cfg.vpu_module).resolve())
    master = (cfg.project_dir/cfg.master_dss).resolve()

    # The supplied runner intentionally reads loadshape / EVAC / pvshape profiles.
    # EVDC/V2G CSVs may participate in total forecasting, but storage dispatch is neutral here.
    load_profiles, evac_profiles, pv_profiles = vpu.load_profile_data(str(generated_dir))
    runner = vpu.OpenDSSVpuRunner(
        master_dss=str(master), selected_buses=target_buses or vpu.SELECT_BUSES,
        load_profiles=load_profiles, evac_profiles=evac_profiles, pv_profiles=pv_profiles,
        scenario_id=4, storage_base_kw=7.0,
    )

    start = LOOK_BACK_DAYS*SLOTS_PER_DAY
    stop = cfg.scenario_days*SLOTS_PER_DAY
    rows = []
    empty_storage = {}  # no V2G control when deciding whether VPP support is needed
    for t in range(start, stop):
        selected, phases, metrics = runner.solve_step_with_metrics(
            t_idx=t, storage_multiplier=empty_storage,
            v_min=cfg.vpp_threshold_pu, v_max=1.05,
        )
        rows.append({
            "t": t, "day": t//SLOTS_PER_DAY + 1, "slot": t%SLOTS_PER_DAY,
            "vpu_min": metrics.vpu_min, "vpu_max": metrics.vpu_max,
            "vpu_mean": metrics.vpu_mean,
            "VPP_request": bool(metrics.vpu_min <= cfg.vpp_threshold_pu),
        })
    return pd.DataFrame(rows)


def run_pipeline(cfg: PipelineConfig, target_buses: Sequence[str] | None = None):
    cfg.project_dir = cfg.project_dir.resolve()
    input_dir = (cfg.project_dir/cfg.csv_inputs_dir).resolve()
    work = (cfg.project_dir/cfg.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    print("[1/7] Build total.csv from csv_inputs")
    files, original = load_aligned_inputs(input_dir, cfg.patterns)
    total = save_total(original, work/"total.csv")
    print(f"  files={len(files)}, samples={len(total)}, complete_days={len(total)//SLOTS_PER_DAY}")

    print("[2/7] Train XGBoost on original total.csv")
    model = train_xgboost(total, cfg.random_seed)

    print(f"[3/7] Bootstrap complete source days -> {cfg.scenario_days} days")
    scenario, chosen = bootstrap_complete_days(original, cfg.scenario_days, cfg.random_seed)
    scenario_total = scenario.sum(axis=1)
    pd.DataFrame({"scenario_total": scenario_total}).to_csv(work/"total_365_sampled.csv", index=False)
    pd.DataFrame({"scenario_day": np.arange(1,cfg.scenario_days+1), "source_day": chosen+1}).to_csv(work/"sampled_day_map.csv", index=False)

    print("[4/7] Forecast day 15..365 using previous 14 days")
    forecast = forecast_next_days(model, scenario_total)
    pd.DataFrame({"sampled_total": scenario_total, "forecast_total": forecast}).to_csv(work/"total_365_forecast.csv", index=False)

    print("[5/7] Reconstruct forecast OpenDSS profiles")
    forecast_profiles = redistribute_forecast(scenario, forecast)
    # First 14 days remain sampled seed profiles; day 15 onward is scaled to forecast total.
    generated = work/"csv_inputs_365_forecast"
    write_profiles(files, forecast_profiles, generated)

    print("[6/7] OpenDSS Vpu -> VPP True/False")
    df = run_opendss_vpp(cfg, generated, target_buses)
    df.to_csv(work/"VPP_request_detailed.csv", index=False)
    # Main/RL notebooks commonly need a one-column request file.
    df[["VPP_request"]].astype(int).to_csv(work/"VPP_request.csv", index=False, header=False)

    print("[7/7] Done")
    for name in ["total.csv", "sampled_day_map.csv", "total_365_forecast.csv", "VPP_request.csv", "VPP_request_detailed.csv"]:
        print(" ", work/name)
    return df
