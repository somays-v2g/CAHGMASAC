from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import opendssdirect as dss


MASTER_DSS = r"./master_with_ev_pv_v2g.dss"
BASE_DIR = os.path.dirname(MASTER_DSS)
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "csv_inputs"))

DEF_V2G_DIR_PATH = r"./sac/def_output"
SAC_V2G_DIR_PATH = r"./sac/v2g_output"
SAC_V2H_DIR_PATH = r"./sac/v2h_output"
TD3_V2G_DIR_PATH = r"./td3/v2g_output"
TD3_V2H_DIR_PATH = r"./td3/v2h_output"
SAC_DEF_DIR_PATH = r"./dsac/v2g_output"
TD3_DEF_DIR_PATH = r"./dtd3/v2h_output"

INTERVAL_H = 0.25

SELECT_BUSES = [
 "1.1","2.2","4.3","5.3","6.3","7.1","9.1","10.1","11.1","12.2",
 "16.3","17.3","19.1","20.1","22.2","24.3","28.1","29.1","30.3","31.3",
 "32.3","33.1","34.3","35.1.2","37.1","38.2","39.2","41.3","42.1","43.2",
 "45.1","46.1","47","48","49.1","49.2","49.3","50.3","51.1","52.1",
 "53.1","55.1","56.2","58.2","59.2","60.1","62.3","63.1","64.2","65.1.2",
 "65.2.3","65.3.1","66.3","68.1","69.1","70.1","71.1","73.3","74.3","75.3",
 "76.1.2","76.2.3","76.3.1","77.2","79.1","80.2","82.1","83.3","84.3","85.3",
 "86.2","87.2","88.1","90.2","92.3","94.1","95.2","96.2","98.1","99.2",
 "100.3","102.3","103.3","104.3","106.2","107.2","109.1","111.1","112.1","113.1",
 "114.1"
]

def _token(fname: str, prefix_regex: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    return re.sub(prefix_regex, "", base, flags=re.I).lower()


def read_ratio_series(path: str) -> np.ndarray:
    path = os.path.abspath(os.fspath(path))

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Profile CSV does not exist: {path}")

    values = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for line_no, row in enumerate(reader, start=1):
                if not row:
                    continue

                token = row[0].strip()
                if token == "":
                    continue

                try:
                    values.append(float(token))
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value in profile CSV: {path} "
                        f"(line {line_no}, first column={token!r})"
                    ) from exc
    except OSError as exc:
        raise OSError(f"Failed to read profile CSV: {path}: {exc}") from exc

    if not values:
        raise ValueError(f"Profile CSV has no numeric data in its first column: {path}")

    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def load_profile_data(data_dir: str = DATA_DIR) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    """Load load, EVAC and PV profile CSVs once."""
    pat_load = os.path.join(
        data_dir, "[lL][oO][aA][dD][sS][hH][aA][pP][eE]_*.c[sS][vV]"
    )
    pat_evac = os.path.join(data_dir, "[eE][vV][aA][cC]_*.c[sS][vV]")
    pat_pv = os.path.join(
        data_dir, "[pP][vV][sS][hH][aA][pP][eE]_*.c[sS][vV]"
    )

    load_map = {_token(path, r"^loadshape_"): path for path in glob.glob(pat_load)}
    evac_map = {
        os.path.splitext(os.path.basename(path))[0].lower(): path
        for path in glob.glob(pat_evac)
    }
    pv_map = {_token(path, r"^pvshape_"): path for path in glob.glob(pat_pv)}

    load_profiles = {name: read_ratio_series(path) for name, path in load_map.items()}
    evac_profiles = {name: read_ratio_series(path) for name, path in evac_map.items()}
    pv_profiles = {name: read_ratio_series(path) for name, path in pv_map.items()}

    if not load_profiles and not evac_profiles and not pv_profiles:
        raise FileNotFoundError(f"No profile CSV files were found under: {data_dir}")

    return load_profiles, evac_profiles, pv_profiles


@dataclass(frozen=True)
class VoltageMetrics:
    voltage_deviation: float
    violation_rate: float
    vpu_min: float
    vpu_max: float
    vpu_mean: float


class OpenDSSVpuRunner:

    def __init__(
        self,
        master_dss: str = MASTER_DSS,
        selected_buses: Sequence[str] = SELECT_BUSES,
        data_dir: Optional[str] = None,
        load_profiles: Optional[Mapping[str, np.ndarray]] = None,
        evac_profiles: Optional[Mapping[str, np.ndarray]] = None,
        pv_profiles: Optional[Mapping[str, np.ndarray]] = None,
        scenario_id: int = 4,
        storage_base_kw: float = 7.0,
    ) -> None:
        self.master_dss = os.path.abspath(master_dss)
        self.selected_buses = tuple(str(x).strip() for x in selected_buses)
        self.scenario_id = int(scenario_id)
        self.storage_base_kw = float(storage_base_kw)
        if self.storage_base_kw <= 0:
            raise ValueError("storage_base_kw must be positive")

        if not os.path.isfile(self.master_dss):
            raise FileNotFoundError(f"MASTER_DSS does not exist: {self.master_dss}")

        if load_profiles is None or evac_profiles is None or pv_profiles is None:
            resolved_data_dir = data_dir or os.path.join(
                os.path.dirname(self.master_dss), "csv_inputs"
            )
            loaded_load, loaded_evac, loaded_pv = load_profile_data(resolved_data_dir)
            load_profiles = loaded_load if load_profiles is None else load_profiles
            evac_profiles = loaded_evac if evac_profiles is None else evac_profiles
            pv_profiles = loaded_pv if pv_profiles is None else pv_profiles

        self.load_profiles = {
            str(k).lower(): np.asarray(v, dtype=float) for k, v in load_profiles.items()
        }
        self.evac_profiles = {
            str(k).lower(): np.asarray(v, dtype=float) for k, v in evac_profiles.items()
        }
        self.pv_profiles = {
            str(k).lower(): np.asarray(v, dtype=float) for k, v in pv_profiles.items()
        }

        self._compile_and_cache()

    def _compile_and_cache(self) -> None:
        dss.Basic.ClearAll()
        dss.Text.Command(f"compile [{self.master_dss}]")
        dss.Text.Command("set controlmode=static")
        dss.Text.Command("set mode=snap")
        dss.Text.Command("set loadmult=1.0")
        self._raise_on_dss_error("compile")

        self.load_names = tuple(dss.Loads.AllNames())
        self.pv_names = tuple(dss.PVsystems.AllNames())
        self.storage_names = tuple(dss.Storages.AllNames())

        self.load_name_lookup = {name.lower(): name for name in self.load_names}
        self.pv_name_lookup = {name.lower(): name for name in self.pv_names}
        self.storage_name_lookup = {name.lower(): name for name in self.storage_names}

        self.base_load_kw: Dict[str, float] = {}
        self.base_load_kvar: Dict[str, float] = {}
        for actual_name in self.load_names:
            dss.Loads.Name(actual_name)
            dss.Loads.Daily("")
            dss.Loads.Yearly("")
            dss.Loads.Duty("")
            dss.Loads.Growth("")
            key = actual_name.lower()
            self.base_load_kw[key] = float(dss.Loads.kW())
            self.base_load_kvar[key] = float(dss.Loads.kvar())

        self.base_pv_pmpp: Dict[str, float] = {}
        for actual_name in self.pv_names:
            dss.PVsystems.Name(actual_name)
            self.base_pv_pmpp[actual_name.lower()] = float(dss.PVsystems.Pmpp())

        self.evac_elements = tuple(
            sorted(set(self.evac_profiles).intersection(self.load_name_lookup))
        )

        # Avoid applying a normal load profile and an EVAC profile twice.
        self.normal_load_keys = tuple(
            key for key in self.load_name_lookup if key not in set(self.evac_elements)
        )

        self._selected_bus_cache = self._build_selected_bus_cache(self.selected_buses)

        for actual_name in self.storage_names:
            dss.Storages.Name(actual_name)
            dss.CktElement.Enabled(1)
            self._set_active_storage_kw(actual_name, 0.0, 0.0)

        if self.evac_profiles and not self.evac_elements:
            print(
                "[WARN] No EVAC CSV profile key matched an OpenDSS Load name. "
                f"Example profile keys={list(self.evac_profiles)[:5]}"
            )

    @staticmethod
    def _parse_bus_nodes(value: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
        parts = str(value).strip().split(".")
        bus_name = parts[0]
        nodes = []
        for item in parts[1:]:
            try:
                nodes.append(int(item))
            except ValueError:
                continue
        return bus_name, tuple(nodes) if nodes else None

    def _build_selected_bus_cache(
        self, selected_buses: Iterable[str]
    ) -> Tuple[Tuple[str, str, Tuple[int, ...]], ...]:
        cache = []
        for raw in selected_buses:
            bus_name, requested_nodes = self._parse_bus_nodes(raw)
            dss.Circuit.SetActiveBus(bus_name)
            available_nodes = tuple(int(x) for x in dss.Bus.Nodes())
            if not available_nodes:
                print(f"[WARN] OpenDSS bus not found or has no nodes: {bus_name}")
                continue

            if requested_nodes is None:
                selected_nodes = available_nodes
            else:
                available_set = set(available_nodes)
                selected_nodes = tuple(n for n in requested_nodes if n in available_set)

            if selected_nodes:
                cache.append((raw, bus_name, selected_nodes))
            else:
                print(f"[WARN] Requested node not present: {raw}")

        if not cache:
            raise ValueError("None of SELECT_BUSES could be resolved in the compiled circuit.")
        return tuple(cache)

    @staticmethod
    def _profile_value(profile: np.ndarray, t_idx: int, name: str) -> float:
        if t_idx < 0 or t_idx >= len(profile):
            raise IndexError(
                f"t_idx={t_idx} is outside profile '{name}' length={len(profile)}"
            )
        return float(profile[t_idx])

    def _apply_load_and_pv_profiles(self, t_idx: int) -> None:

        for key in self.normal_load_keys:
            actual_name = self.load_name_lookup[key]
            profile = self.load_profiles.get(key)
            mult = (
                self._profile_value(profile, t_idx, key)
                if profile is not None
                else 1.0
            )
            dss.Loads.Name(actual_name)
            dss.Loads.kW(self.base_load_kw[key] * mult)
            dss.Loads.kvar(self.base_load_kvar[key] * mult)

        if self.scenario_id >= 2:
            for key in self.evac_elements:
                actual_name = self.load_name_lookup[key]
                profile = self.evac_profiles[key]
                mult = self._profile_value(profile, t_idx, key)
                dss.Loads.Name(actual_name)
                dss.Loads.kW(self.base_load_kw[key] * mult)
                dss.Loads.kvar(self.base_load_kvar[key] * mult)
        else:
            for key in self.evac_elements:
                dss.Loads.Name(self.load_name_lookup[key])
                dss.Loads.kW(0.0)
                dss.Loads.kvar(0.0)

        pv_enabled = self.scenario_id in (3, 4)
        for key, actual_name in self.pv_name_lookup.items():
            profile = self.pv_profiles.get(key)
            if pv_enabled and profile is not None:
                mult = self._profile_value(profile, t_idx, key)
                pmpp = self.base_pv_pmpp[key] * mult
            else:
                pmpp = 0.0
            dss.PVsystems.Name(actual_name)
            dss.PVsystems.Pmpp(pmpp)

    @staticmethod
    def _set_active_storage_kw(actual_name: str, p_kw: float, q_kvar: float) -> None:
        if p_kw > 0.0:
            state = "Discharging"
            magnitude_kw = p_kw
        elif p_kw < 0.0:
            state = "Charging"
            magnitude_kw = abs(p_kw)
        else:
            state = "Idling"
            magnitude_kw = 0.0
            q_kvar = 0.0

        dss.Text.Command(
            f"edit storage.{actual_name} state={state} "
            f"kW={magnitude_kw:.12g} kvar={q_kvar:.12g}"
        )

    def _apply_storage_multiplier(
        self,
        storage_multiplier: Mapping[str, float],
        storage_kvar: Optional[Mapping[str, float]] = None,
    ) -> None:
        normalized_p = {str(k).lower(): float(v) for k, v in storage_multiplier.items()}
        normalized_q = (
            {str(k).lower(): float(v) for k, v in storage_kvar.items()}
            if storage_kvar is not None
            else {}
        )

        unknown = set(normalized_p).difference(self.storage_name_lookup)
        if unknown:
            raise KeyError(
                "Storage names were not found in the compiled OpenDSS circuit: "
                + ", ".join(sorted(unknown)[:10])
            )

        

        for key, actual_name in self.storage_name_lookup.items():                       
            dss.Storages.Name(actual_name)
            dss.CktElement.Enabled(1)
            self._set_active_storage_kw(
                actual_name,
                normalized_p.get(key, 0.0) * self.storage_base_kw,
                normalized_q.get(key, 0.0),
            )

    def _read_selected_vpu(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        selected: Dict[str, float] = {}
        phases: Dict[str, float] = {}

        for raw, bus_name, selected_nodes in self._selected_bus_cache:
            dss.Circuit.SetActiveBus(bus_name)
            bus_nodes = tuple(int(x) for x in dss.Bus.Nodes())
            pu_voltage = dss.Bus.PuVoltage()

            if not bus_nodes or not pu_voltage:
                selected[raw] = float("nan")
                continue

            node_to_vpu: Dict[int, float] = {}
            for i, node in enumerate(bus_nodes):
                re_val = float(pu_voltage[2 * i])
                im_val = float(pu_voltage[2 * i + 1])
                node_to_vpu[node] = float(np.hypot(re_val, im_val))

            values = []
            for node in selected_nodes:
                value = node_to_vpu.get(node)
                if value is not None:
                    phases[f"{bus_name}.{node}"] = value
                    values.append(value)

            selected[raw] = float(max(values)) if values else float("nan")

        return selected, phases

    @staticmethod
    def calculate_metrics(
        values: Mapping[str, float], v_min: float = 0.95, v_max: float = 1.05
    ) -> VoltageMetrics:
        voltages = np.asarray(
            [float(v) for v in values.values() if np.isfinite(v)], dtype=float
        )
        if voltages.size == 0:
            return VoltageMetrics(
                voltage_deviation=float("nan"),
                violation_rate=float("nan"),
                vpu_min=float("nan"),
                vpu_max=float("nan"),
                vpu_mean=float("nan"),
            )

        violations = int(np.any(voltages <= v_min))
        return VoltageMetrics(
            voltage_deviation=float(np.mean(np.abs(voltages - 1.0))),
            violation_rate=violations,
            vpu_min=float(np.min(voltages)),
            vpu_max=float(np.max(voltages)),
            vpu_mean=float(np.mean(voltages)),
        )

    def solve_step_kw(
        self,
        t_idx: int,
        storage_power_kw: Mapping[str, float],
        storage_kvar: Optional[Mapping[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        storage_multiplier = {
            str(name): float(power_kw) / self.storage_base_kw
            for name, power_kw in storage_power_kw.items()
        }
        return self.solve_step(t_idx, storage_multiplier, storage_kvar)

    def solve_step(
        self,
        t_idx: int,
        storage_multiplier: Mapping[str, float],
        storage_kvar: Optional[Mapping[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        
        self._apply_load_and_pv_profiles(int(t_idx))
        self._apply_storage_multiplier(storage_multiplier, storage_kvar)

        dss.Solution.Solve()
        self._raise_on_dss_error(f"solve t_idx={t_idx}")
        if not bool(dss.Solution.Converged()):
            raise RuntimeError(f"OpenDSS did not converge at t_idx={t_idx}")

        return self._read_selected_vpu()

    def solve_step_with_metrics(
        self,
        t_idx: int,
        storage_multiplier: Mapping[str, float],
        storage_kvar: Optional[Mapping[str, float]] = None,
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> Tuple[Dict[str, float], Dict[str, float], VoltageMetrics]:
        selected, phases = self.solve_step(t_idx, storage_multiplier, storage_kvar)
        metrics = self.calculate_metrics(selected, v_min=v_min, v_max=v_max)
        return selected, phases, metrics

    def reset_episode(self) -> None:
        for actual_name in self.storage_names:
            dss.Storages.Name(actual_name)
            dss.CktElement.Enabled(1)
            self._set_active_storage_kw(actual_name, 0.0, 0.0)

    def recompile(self) -> None:
        self._compile_and_cache()

    @staticmethod
    def _raise_on_dss_error(context: str) -> None:
        error_number = int(dss.Error.Number())
        if error_number != 0:
            raise RuntimeError(
                f"OpenDSS error during {context}: {dss.Error.Description()}"
            )


# ========= Preferred convenience API =========
def create_vpu_runner(
    master_dss: str = MASTER_DSS,
    selected_buses: Sequence[str] = SELECT_BUSES,
    data_dir: Optional[str] = None,
) -> OpenDSSVpuRunner:
    return OpenDSSVpuRunner(
        master_dss=master_dss,
        selected_buses=selected_buses,
        data_dir=data_dir,
        scenario_id=4,
        storage_base_kw=7.0,
    )


def get_vpu_fast(
    runner: OpenDSSVpuRunner,
    t_idx: int,
    storage_multiplier: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    return runner.solve_step(t_idx=t_idx, storage_multiplier=storage_multiplier)


def _legacy_v2g_directory(algo: str) -> str:
    return {
        "hsac": SAC_V2G_DIR_PATH,
        "sac": SAC_V2H_DIR_PATH,
        "htd3": TD3_V2G_DIR_PATH,
        "td3": TD3_V2H_DIR_PATH,
        "dsac": SAC_DEF_DIR_PATH,
        "dtd3": TD3_DEF_DIR_PATH,
    }.get(str(algo).lower(), DEF_V2G_DIR_PATH)


def get_vpu_legacy(
    t_idx: int,
    policy: str,
    algo: str,
    runner: Optional[OpenDSSVpuRunner] = None,
    normalized_csv_values: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    
    del policy  
    if runner is None:
        runner = create_vpu_runner()

    directory = _legacy_v2g_directory(algo)
    pattern = os.path.join(directory, "[vV][2][gG]_*.c[sS][vV]")
    storage_multiplier: Dict[str, float] = {}

    for path in glob.glob(pattern):
        storage_name = _token(path, r"")
        profile = read_ratio_series(path)
        value = OpenDSSVpuRunner._profile_value(profile, int(t_idx), storage_name)
        storage_multiplier[storage_name] = (
            value if normalized_csv_values else value / runner.storage_base_kw
        )

    return runner.solve_step(int(t_idx), storage_multiplier)


def get_vpu(t_idx: int, policy: str, algo: str):
    return get_vpu_legacy(t_idx=t_idx, policy=policy, algo=algo)
