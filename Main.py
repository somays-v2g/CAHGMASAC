import os, sys, json, time, gc
import random as _pyrandom
from time import perf_counter as _pc, process_time as _cpu
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from investigate_Vpu import create_vpu_runner
from gen_ev_model import EVCusConfig, generate_ev
from agent_env import V2GHomeEnv, TripConfig, load_prices_from_csv, load_ev_to_bus_map
from agent_policy import MultiAgentSAC


RUN_DIR = ""
PROJECT_DIR = ""
OUT_BASE = "results"
VPP_FORECAST_DIR = "vpp_forecast_output"

SEED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
COND_INDEX = -1
RESUME = False
N_EVS = 50
BUFFER_SIZE = 200000
TRAIN_DAYS = 330
WARMUP_FRAC = 0.2
EVAL_DAYS = 30
EVAL_SEED = 100
EVAL_P_RANGE = (0.3, 1.0)
WARMUP_EPS = 0.3
WARMUP_NOISE = 0.1
VPP_THRESHOLD = 0.952
DEDUP = True
RUN_AGGREGATE = False
LAMBDA_DEG = 1.0
P_BASE = 0.8
TD3_NOISE = 5e-8
SCALING_EVS = (100, 200, 300, 400, 500)
SCALING_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
SCALING_ALGOS = ("hsac",)

T2_MODE = "participation"
T2_ALGOS = ("hsac",)                 
T2_NOV2X_MODE = "charge_only"        
T2_FIXED_EVAL_P = True               
ONLY_STUDIES = ("t2_",)              


def _apply_overrides(ns):
    import ast as _ast
    ov = {}
    if os.environ.get("MAIN_PARAMS"):
        ov.update(json.loads(os.environ["MAIN_PARAMS"]))
    if ns.get("__name__") == "__main__":
        for arg in sys.argv[1:]:
            if "=" not in arg:
                sys.exit(f"[Main] : {arg!r}")
            k, v = arg.split("=", 1)
            try:
                ov[k.strip()] = _ast.literal_eval(v)
            except (ValueError, SyntaxError):
                ov[k.strip()] = v
    for k, v in ov.items():
        if k not in ns:
            sys.exit(f"[Main] Error: {k}")
        ns[k] = tuple(v) if isinstance(ns[k], tuple) and isinstance(v, list) else v
    return ov


_OVERRIDES = _apply_overrides(globals())

if T2_MODE == "participation":
    _T2_400, _T2_500 = (500, 0.8), (500, 1.0)
else:
    _T2_400, _T2_500 = (400, 1.0), (500, 1.0)
# (study, deg_w, lambda_vdev, p, rho, tariff, n_evs)
T2_STUDIES = [
    ("t2_noV2X",   LAMBDA_DEG, 0.2, _T2_400[1], 0.4, "CA", _T2_400[0]),  
    ("t2_nopen",   LAMBDA_DEG, 0.0, _T2_400[1], 0.4, "CA", _T2_400[0]),  
    ("t2_ours400", LAMBDA_DEG, 0.2, _T2_400[1], 0.4, "CA", _T2_400[0]),  
    ("t2_ours500", LAMBDA_DEG, 0.2, _T2_500[1], 0.4, "CA", _T2_500[0]), 
]
V_MIN, V_MAX = 0.95, 1.05

ALL_STUDIES = (
    [("degradation", d, 0.2, P_BASE, 0.4, "CA") for d in (0.0, 1.0, 2.0, 5.0)]
    + [("w3", LAMBDA_DEG, w3, P_BASE, 0.4, "CA") for w3 in (0.2, 0.5, 1.0, 1.5, 2.0)]
    + [("participation", LAMBDA_DEG, 0.2, p, 0.4, "CA") for p in (1.0, 0.8, 0.5, 0.3)]
    + [("rho", LAMBDA_DEG, 0.2, P_BASE, r, "CA") for r in (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0)]
    + [("tariff", LAMBDA_DEG, 0.2, P_BASE, 0.4, t) for t in ("CA", "TX")]
    + [("scaling", LAMBDA_DEG, 0.2, P_BASE, 0.4, "CA", n) for n in SCALING_EVS]
    + T2_STUDIES                      
)


def cond_tag(cond):
    study, d, w3, p, r, t = cond[:6]
    f = lambda x: f"{x:g}".replace(".", "p")
    s = f"{study}_degw_{f(d)}_vdevw_{f(w3)}_p_{f(p)}_rho_{f(r)}_{t}"
    return s + (f"_n{cond[6]}" if len(cond) > 6 else "")
# <<< PARAMS

PROJECT_DIR = Path(PROJECT_DIR or os.getcwd()).resolve()
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

VPP_DAYS = int(TRAIN_DAYS) + int(EVAL_DAYS)
VPP_REQUEST_PATH = str(PROJECT_DIR / VPP_FORECAST_DIR / "VPP_request.csv")

def load_vpp(path):
    import pandas as _pd
    df = _pd.read_csv(path, header=None)
    if str(df.iloc[0, 0]).strip().lower() in ("true", "false", "1", "0", "1.0", "0.0"):
        s = df.iloc[:, 0]
    else:
        df = _pd.read_csv(path)
        s = df["VPP_request"] if "VPP_request" in df.columns else df.iloc[:, -1]
    return [str(x).strip().lower() in ("true", "1", "1.0") for x in s.tolist()]

def save_vpp_plain(path, flags):
    tmp = Path(f"{path}.tmp{os.getpid()}")
    tmp.write_text("".join(f"{bool(v)}\n" for v in flags), encoding="utf-8")
    os.replace(tmp, path)

if not Path(VPP_REQUEST_PATH).exists():
    import vpp_forecast as VF
    VF.ensure_vpp_request(VF.PipelineConfig(project_dir=PROJECT_DIR, work_dir=Path(VPP_FORECAST_DIR),
                                            vpp_threshold_pu=float(VPP_THRESHOLD)),
                          n_forecast_days=VPP_DAYS)
    save_vpp_plain(VPP_REQUEST_PATH, load_vpp(VPP_REQUEST_PATH))

vpp_request = load_vpp(VPP_REQUEST_PATH)

bus_per_home_path = str(PROJECT_DIR / "v2ghome_bus.csv")
price_path = str(PROJECT_DIR / "csv_inputs" / "price_t.csv")
MASTER_DSS_PATH = str(PROJECT_DIR / "master_with_ev_pv_v2g.dss")

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)
_CUDA = torch.cuda.is_available()
_CUDA_DEVS = list(range(torch.cuda.device_count())) if _CUDA else []


def _seed_torch(s):
    torch.manual_seed(int(s))
    if _CUDA:
        torch.cuda.manual_seed_all(int(s))


class _AlgoRNG:
    def __init__(self, seed):
        with torch.random.fork_rng(devices=_CUDA_DEVS):
            _seed_torch(seed)
            self.cpu = torch.get_rng_state()
            self.cuda = [torch.cuda.get_rng_state(d) for d in _CUDA_DEVS]

    def __enter__(self):
        self._saved_cpu = torch.get_rng_state()
        self._saved_cuda = [torch.cuda.get_rng_state(d) for d in _CUDA_DEVS]
        torch.set_rng_state(self.cpu)
        for d, s in zip(_CUDA_DEVS, self.cuda):
            torch.cuda.set_rng_state(s, d)
        return self

    def __exit__(self, *exc):
        self.cpu = torch.get_rng_state()
        self.cuda = [torch.cuda.get_rng_state(d) for d in _CUDA_DEVS]
        torch.set_rng_state(self._saved_cpu)
        for d, s in zip(_CUDA_DEVS, self._saved_cuda):
            torch.cuda.set_rng_state(s, d)
        return False


def _disable_td3_diagnostics(agent):
    agent.diag_enabled = False
    agent._diag_pending_print = False
    for role_agent in agent.role_algo.values():
        role_agent.diag_actor_enabled = False
        role_agent.diag_actor_limit = 0
    return agent


trip_cfg = TripConfig()
ev_cfg = EVCusConfig(n_evs=N_EVS)

if SEED is None:
    seeds = list(range(10))
    _PTAG = "_allseeds"
elif isinstance(SEED, (list, tuple)):
    seeds = [int(s) for s in SEED]
    _PTAG = f"_seed{seeds[0]}" if len(seeds) == 1 else "_seeds" + "-".join(str(s) for s in seeds)
else:
    seeds = [int(SEED)]
    _PTAG = f"_seed{seeds[0]}"
COND_INDEX = int(COND_INDEX)
if COND_INDEX > 0:
    _PTAG += f"_c{COND_INDEX}"

TD3_EXPL_NOISE = float(TD3_NOISE)
NOISE_TAG = "tdn" + f"{TD3_EXPL_NOISE:.0e}".replace("-0", "-").replace("+0", "")

STANDALONE = COND_INDEX <= 0          
OUT_ROOT = (Path(RUN_DIR) if RUN_DIR else PROJECT_DIR / OUT_BASE / f"run_{datetime.now():%Y%m%d_%H%M%S}").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
_CUR_DIR = OUT_ROOT


def _ensure_workdir():
    if Path.cwd() != _CUR_DIR:
        os.chdir(_CUR_DIR)


def _use_seed_dir(seed):
    global _CUR_DIR
    _CUR_DIR = OUT_ROOT / f"seed{int(seed)}"
    _CUR_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_workdir()


def _stag(seed, cond_i):    
    return f"_seed{int(seed)}_c{int(cond_i)}"


_ensure_workdir()

_vpp_copy = OUT_ROOT / "VPP_request.csv"
if not _vpp_copy.exists():
    _tmp = OUT_ROOT / f"VPP_request.csv.tmp{os.getpid()}"
    _tmp.write_text("".join(f"{bool(v)}\n" for v in vpp_request), encoding="utf-8")
    os.replace(_tmp, _vpp_copy)

t_DAYS = int(TRAIN_DAYS)
e_DAYS = int(EVAL_DAYS)
DAYS = t_DAYS + e_DAYS
warmup = int(round(t_DAYS * WARMUP_FRAC))

reward_w = 1
GRID_PENALTY_SCALE = float(os.environ.get("V2G_GRID_SCALE", "0.1"))
lambda_vio = float(os.environ.get("V2G_W2", "2"))
vdev_max = 0.05187
vdev_min = 0.00013

_T_PREP0 = _pc()
pr = load_prices_from_csv(price_path) * 0.55
if pr.ndim == 2 and pr.shape[1] == 1:
    pr = pr[:, 0]


def tariff_setting(loca):
    if loca == "CA":
        return np.concatenate((pr, pr), axis=0), 2
    return np.concatenate((pr / 0.55 * 0.14, pr / 0.55 * 0.14), axis=0), 1


STEPS_PER_DAY = 96
peak_start = 16 * 4
peak_end = 21 * 4

load_ev_to_bus_map(bus_per_home_path)

PREP_S = _pc() - _T_PREP0

EVAL_P_SCHEDULE = np.random.default_rng([int(EVAL_SEED), 777]).uniform(*EVAL_P_RANGE, e_DAYS).round(2).tolist()
EVAL_FLEET = generate_ev(ev_cfg, rng=np.random.default_rng([int(EVAL_SEED), 20240001]))


def sample_participating_evs(ev_initial, participation_rate, rng):
    ev_ids = ev_initial["ev_id"]
    n_part = int(round(len(ev_ids) * participation_rate))
    if n_part >= len(ev_ids):
        return set(ev_ids)
    if n_part <= 0:
        return set()
    return set(rng.choice(ev_ids, size=n_part, replace=False))


def heuristic_demo_actions(obs_by_agent):
    actions = {}
    for aid, obs in obs_by_agent.items():
        if aid == "DERMS":
            actions[aid] = 0.0
            continue
        connected = bool(obs.get("connectivity", False))
        current_soc = float(obs.get("current_soc", obs.get("soc_after_trip", 1.0)))
        charge_thres = float(obs.get("charge_thres", 0.0))
        v2g_min_thres = float(obs.get("V2G_min_thres", 0.0))
        vpp_req = bool(obs.get("vpp_req", False))        
        if not connected:
            action = 0.25
        elif current_soc <= charge_thres:
            action = 0.75
        elif vpp_req and current_soc > v2g_min_thres:
            action = -0.75
        elif current_soc > v2g_min_thres:
            action = -0.25
        else:
            action = 0.25
        actions[aid] = float(action)
    return actions


def warmup_explore_actions(obs_by_agent, seed, d_idx, p_idx):
    base = heuristic_demo_actions(obs_by_agent)
    ids = sorted(base)
    rng = np.random.default_rng([int(seed), 4242, int(d_idx), int(p_idx)])
    u = rng.random(len(ids))
    rand_a = rng.uniform(-1.0, 1.0, len(ids))
    noise = rng.normal(0.0, float(WARMUP_NOISE), len(ids)) if WARMUP_NOISE > 0 else np.zeros(len(ids))
    out = {}
    for i, aid in enumerate(ids):
        if u[i] < WARMUP_EPS:
            out[aid] = float(rand_a[i])
        else:
            out[aid] = float(np.clip(base[aid] + noise[i], -1.0, 1.0))
    return out


def apply_participation_to_action(action_by_agent, obs_by_agent, participant_ids):
    final_action = {}
    for ev_id in obs_by_agent.keys():
        if ev_id in participant_ids:
            final_action[ev_id] = action_by_agent[ev_id]
        else:
            final_action[ev_id] = 0.0
            obs_by_agent[ev_id]["current_act"] = "Idle"
    return final_action, obs_by_agent


def _ev_sum(amount_by_agent):
    return float(sum(float(v) for k, v in amount_by_agent.items() if str(k).startswith("EV_")))


hsac_cfg = {"algo": "hsac", "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
            "buffer_size": int(BUFFER_SIZE), "batch_size": 256, "updates_per_step": 1, "steps_hint": 96}
sac_cfg = {"algo": "sac", "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
           "buffer_size": int(BUFFER_SIZE), "batch_size": 256, "updates_per_step": 1, "steps_hint": 96}
htd3_cfg = {"algo": "htd3", "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4, "critic_lr": 3e-4,
            "buffer_size": int(BUFFER_SIZE), "batch_size": 256, "learning_starts": 1000, "updates_per_step": 1,
            "policy_noise": 0.2, "noise_clip": 0.5, "policy_delay": 2}
td3_cfg = {"algo": "td3", "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4, "critic_lr": 3e-4,
           "buffer_size": int(BUFFER_SIZE), "batch_size": 256, "learning_starts": 1000, "updates_per_step": 1,
           "policy_noise": 0.2, "noise_clip": 0.5, "policy_delay": 2}
desac_cfg = {"algo": "dsac", "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
             "buffer_size": int(BUFFER_SIZE), "batch_size": 256, "updates_per_step": 1, "steps_hint": 96}
detd3_cfg = {"algo": "dtd3", "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
             "buffer_size": int(BUFFER_SIZE), "batch_size": 256, "learning_starts": 1000, "updates_per_step": 1,
             "steps_hint": 96}

ALGO_SPECS = {
    "hsac": dict(cfg=hsac_cfg,  hg=True,  td3=False, policy="Gaussian", init=1001, prefix="sac_hg", name="CAHG-MASAC"),
    "sac":  dict(cfg=sac_cfg,   hg=False, td3=False, policy="Gaussian", init=1001, prefix="sac_g",  name="CA-MASAC"),
    "htd3": dict(cfg=htd3_cfg,  hg=True,  td3=True,  policy="Gaussian", init=2001, prefix="td3_hg", name="CAHG-MATD3"),
    "td3":  dict(cfg=td3_cfg,   hg=False, td3=True,  policy="Gaussian", init=2001, prefix="td3_g",  name="CA-MATD3"),
    "dsac": dict(cfg=desac_cfg, hg=False, td3=False, policy="Default",  init=3001, prefix="de_sac", name="MASAC"),
    "dtd3": dict(cfg=detd3_cfg, hg=False, td3=True,  policy="Default",  init=4001, prefix="de_td3", name="MATD3"),
}
ALGOS = tuple(ALGO_SPECS)

_N_COND_ALL = len(ALL_STUDIES)
_COND_LIST = list(enumerate(ALL_STUDIES, 1))
if COND_INDEX > 0:
    _COND_LIST = [_COND_LIST[COND_INDEX - 1]]
elif ONLY_STUDIES:
    _COND_LIST = [(i, c) for i, c in _COND_LIST if str(c[0]).startswith(tuple(ONLY_STUDIES))]

_ALIASES, _RUN_LIST, _owner = [], [], {}
for _i, _c in _COND_LIST:
    _k = tuple(_c[1:])
    if DEDUP and STANDALONE and _c[0] != "scaling" and not str(_c[0]).startswith("t2_") and _k in _owner:
        _ALIASES.append((_i, _owner[_k]))
    else:
        _owner.setdefault(_k, _i)
        _RUN_LIST.append((_i, _c))


if STANDALONE and not (OUT_ROOT / "run_config.json").exists():
    (OUT_ROOT / "run_config.json").write_text(json.dumps({
        "run_id": OUT_ROOT.name, "td3_noise": TD3_EXPL_NOISE,
        "studies": ",".join(dict.fromkeys(c[0] for _, c in _COND_LIST)),
        "n_evs": N_EVS, "train_days": t_DAYS, "warmup_frac": float(WARMUP_FRAC), "eval_days": e_DAYS,
        "eval_seed": int(EVAL_SEED), "eval_p_range": ",".join(str(float(x)) for x in EVAL_P_RANGE),
        "warmup_eps": float(WARMUP_EPS), "warmup_noise": float(WARMUP_NOISE), "warmup_vpp_schedule": False,
        "vpp_dir": VPP_FORECAST_DIR, "vpp_threshold": float(VPP_THRESHOLD), "dedup": bool(DEDUP),
        "seeds": seeds, "n_conditions": len(_RUN_LIST),
        "aliases": [[cond_tag(ALL_STUDIES[a - 1]), cond_tag(ALL_STUDIES[o - 1])] for a, o in _ALIASES],
        "notebook": "Main.ipynb (standalone)", "project_dir": str(PROJECT_DIR),
        "started": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

with open(OUT_ROOT / f"run_settings{_PTAG}.json", "w", encoding="utf-8") as _f:
    json.dump({"seeds": seeds, "td3_expl_noise": TD3_EXPL_NOISE, "n_evs": N_EVS, "total_days": DAYS,
               "train_days": t_DAYS, "eval_days": e_DAYS, "warmup_days": warmup, "eval_seed": int(EVAL_SEED),
               "eval_participation_schedule": EVAL_P_SCHEDULE, "conditions": [list(c) for _, c in _COND_LIST],
               "cond_index": COND_INDEX, "vpp_request_path": VPP_REQUEST_PATH,
               "vpp_threshold": float(VPP_THRESHOLD), "dedup_aliases": [list(x) for x in _ALIASES], "vpp_request": [bool(v) for v in vpp_request[:DAYS]],
               "started": datetime.now().isoformat(timespec="seconds")}, _f, indent=2)

_OUT_FILES = {"summary": "all_reward_weight_sensitivity_seed_summary", "eval": "all_eval_summary",
              "soc": "all_soc_trajectory", "daily": "all_daily_action_counts", "daily_reward": "all_daily_rewards",
              "soc_profile": "all_soc_profile", "timing": "all_timing",
              "grid_steps": "all_grid_steps", "grid": "all_grid_summary"}


def _paths(seed, cond_i):
    return {k: f"{v}{_stag(seed, cond_i)}.csv" for k, v in _OUT_FILES.items()}


if not RESUME:
    for _s in seeds:
        for _ci, _ in _COND_LIST:        
            for _f in _paths(_s, _ci).values():
                _p = OUT_ROOT / f"seed{_s}" / _f
                if _p.exists():
                    _p.unlink()
_T_RUN0 = time.time()
_ACTIONS = ("Charging", "V2G", "V2H", "Idle")
_SUM_COLS = {
    "V2H_kWh": "V2H_kWh", "V2G_kWh": "V2G_kWh", "charging_kWh": "charge_kWh",
    "forced_recharge_kWh": "recharge_kWh", "economic_reward": "economic_reward",
    "recharge_cost": "recharge_cost", "degradation_cost": "degradation_cost",
    "degradation_penalty_applied": "degradation_penalty_applied",
    "reward_before_normalization": "reward_before_normalization", "final_reward": "reward_final",
}


def _append_csv(path, df):
    if df is None or len(df) == 0:
        return
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


class _BehaviourAccumulator:
    def __init__(self, stat="train"):
        self.stat = stat
        self.sums = dict.fromkeys(_SUM_COLS, 0.0)
        self.n = 0
        self.soc_sum = 0.0
        self.soc_min = float("inf")
        self.vdev_sum = 0.0
        self.vio_sum = 0.0
        self.actions = {}
        self.last_soc = {}
        self.soc_traj = {}
        self.daily = []

    def consume(self, df, want_daily=True):
        if df is None or df.empty:
            return
        if want_daily:
            for (day, st, pol, vpp), g in df.groupby(["day_idx", "stat", "policy", "vpp_req"], sort=True):
                vc = g["executed_action"].value_counts()
                rec = {"day_idx": int(day), "stat": str(st), "policy": str(pol), "vpp_req": bool(vpp),
                       "n_ev_steps": int(len(g))}
                for a in _ACTIONS:
                    rec[f"n_{a}"] = int(vc.get(a, 0))
                rec["n_ev_charging"] = int(g.loc[g["executed_action"] == "Charging", "agent_id"].nunique())
                for k_out, k_in in (("charging_kWh", "charge_kWh"), ("V2G_kWh", "V2G_kWh"), ("V2H_kWh", "V2H_kWh"),
                                    ("recharge_kWh", "recharge_kWh"), ("degradation_cost", "degradation_cost"),
                                    ("reward_final", "reward_final")):
                    rec[k_out] = float(g[k_in].sum())
                rec["mean_soc_after"] = float(g["soc_after"].mean())
                self.daily.append(rec)
        d = df[(df["policy"] != "Heuristic") & (df["stat"] == self.stat)]
        if d.empty:
            return
        for col, key in _SUM_COLS.items():
            self.sums[col] += float(d[key].sum())
        self.n += int(len(d))
        self.soc_sum += float(d["soc_after"].sum())
        self.soc_min = min(self.soc_min, float(d["soc_after"].min()))
        self.vdev_sum += float(d["voltage_deviation"].sum())
        self.vio_sum += float(d["violation_rate"].sum())
        for a, c in d["executed_action"].value_counts().items():
            self.actions[str(a)] = self.actions.get(str(a), 0) + int(c)
        self.last_soc.update(zip(d["agent_id"], d["soc_after"]))
        for ti, s in d.groupby("t_idx")["soc_after"].agg(["sum", "count"]).iterrows():
            acc = self.soc_traj.setdefault(int(ti), [0.0, 0])
            acc[0] += float(s["sum"]); acc[1] += int(s["count"])

    def summary_row(self, meta):
        row = dict(meta)
        row.update(self.sums)
        n = max(self.n, 1)
        row["mean_SOC"] = self.soc_sum / n if self.n else np.nan
        row["min_SOC"] = self.soc_min if self.n else np.nan
        row["mean_voltage_deviation"] = self.vdev_sum / n if self.n else np.nan
        row["mean_violation_rate"] = self.vio_sum / n if self.n else np.nan
        row["Total_V2X_kWh"] = row["V2H_kWh"] + row["V2G_kWh"]
        row["final_SOC"] = float(np.mean(list(self.last_soc.values()))) if self.last_soc else np.nan
        row["n_ev_steps"] = self.n
        tot = max(sum(self.actions.values()), 1)
        for a in _ACTIONS:
            row[a] = self.actions.get(a, 0) / tot
            row[f"count_{a}"] = self.actions.get(a, 0)
        return row

    def soc_df(self, meta):
        if not self.soc_traj:
            return pd.DataFrame()
        t = sorted(self.soc_traj)
        df = pd.DataFrame({"t_idx": t,
                           "soc_after_mean": [self.soc_traj[k][0] / max(self.soc_traj[k][1], 1) for k in t],
                           "n_ev": [self.soc_traj[k][1] for k in t]})
        for k, v in meta.items():
            df[k] = v
        return df

    def daily_df(self, meta):
        df = pd.DataFrame(self.daily)
        for k, v in meta.items():
            df[k] = v
        return df


_SOC_GROUPS = (("all", None), ("participant", "participant"), ("participant_connected", "part_conn"))


def soc_profile_rows(df, reserve, batt):
    if df.empty:
        return []
    d = df.copy()
    d["part_conn"] = d["participant"] & d["connected"]
    d["soc_after_op"] = (d["soc_after"] - d["recharge_kWh"] / d["agent_id"].map(batt)).clip(lower=0.0)
    d["below_reserve"] = d["soc_after_op"] < d["agent_id"].map(reserve) - 1e-6
    rows = []
    for (day, step, vpp, st), g in d.groupby(["day_idx", "step_of_day", "vpp_req", "stat"], sort=True):
        for name, col in _SOC_GROUPS:
            q = g if col is None else g[g[col]]
            rows.append({"day_idx": int(day), "step_of_day": int(step), "hour": int(step) // 4,
                         "vpp_req": bool(vpp), "stat": str(st), "group": name, "n_ev": int(len(q)),
                         "soc_before_sum": float(q["soc_before"].sum()), "soc_after_sum": float(q["soc_after"].sum()),
                         "soc_after_op_sum": float(q["soc_after_op"].sum()),
                         "soc_after_min": float(q["soc_after_op"].min()) if len(q) else float("nan"),
                         "n_below_reserve": int(q["below_reserve"].sum()),
                         "V2G_kWh": float(q["V2G_kWh"].sum()), "V2H_kWh": float(q["V2H_kWh"].sum()),
                         "charge_kWh": float(q["charge_kWh"].sum())})
    return rows


def _make_env(seed, vpp_incen, runner, deg_weight, ev_cfg_):
    env = V2GHomeEnv(seed_train=[seed], VPP_incentive=vpp_incen, re_w=reward_w, ev_cfg=ev_cfg_,
                     dss_runner=runner, seed_eval=int(EVAL_SEED))
    env.use_batt_deg = deg_weight > 0.0
    env.batt_deg_weight = float(deg_weight)
    env.freeze_r_max = True
    env.grid_penalty_share = "per_agent"
    env.grid_penalty_scale = GRID_PENALTY_SCALE
    env.shared_exogenous = True
    env.resample_fleet_each_reset = False
    return env


_TIMING_KEYS = ("trip_obs", "act", "env_step", "opendss", "buffer", "nn_update", "logging")


def _instrument_runner(runner, T):
    for name in ("solve_step_with_metrics", "reset_episode"):
        f = getattr(runner, name, None)
        if f is None:
            continue

        def wrapped(*a, _f=f, **k):
            t0 = _pc()
            try:
                return _f(*a, **k)
            finally:
                T["opendss"] += _pc() - t0
        setattr(runner, name, wrapped)
    return runner


def _vpu_array(vpu):
    if vpu is None:
        return np.array([])
    if isinstance(vpu, dict):
        vpu = vpu.get("vpu", list(vpu.values()))
    if isinstance(vpu, pd.DataFrame):
        vpu = vpu["vpu"] if "vpu" in vpu.columns else vpu.select_dtypes("number").to_numpy()
    a = pd.to_numeric(pd.Series(np.ravel(np.asarray(vpu, dtype=object))), errors="coerce").dropna()
    return a.to_numpy(dtype=float)


def _grid_step_row(env, vdev, vio):
    v = _vpu_array(getattr(env, "last_vpu", None))
    n_out = int(((v < V_MIN) | (v > V_MAX)).sum()) if v.size else -1
    return {"voltage_deviation": float(vdev), "violation_rate": float(vio),
            "n_bus": int(v.size), "n_bus_violation": n_out,          # 버스 단위 위반 수
            "step_violation": int(n_out > 0) if v.size else int(vio > 0),   # 위반 발생 스텝 여부
            "vpu_min": float(v.min()) if v.size else np.nan, "vpu_max": float(v.max()) if v.size else np.nan,
            "max_abs_dev": float(np.abs(v - 1.0).max()) if v.size else np.nan}


def run_one_seed(seed, cond_i, cond, tag):
    SWEEP_TYPE, BATT_DEG_WEIGHT, lambda_vdev, participation_rate, buffer_frac, loca = cond[:6]
    SCALING = SWEEP_TYPE == "scaling"
    TABLE2 = str(SWEEP_TYPE).startswith("t2_")
    NO_V2X = SWEEP_TYPE == "t2_noV2X"                       
    if SWEEP_TYPE == "t2_nopen":
        lambda_vdev = 0.0
        lam_vio_c = 0.0
    else:
        lam_vio_c = lambda_vio
    if SWEEP_TYPE == "t2_nopen":
        assert lambda_vdev == 0.0 and lam_vio_c == 0.0
    print(f"[grid penalty] study={SWEEP_TYPE}, lambda_vdev={lambda_vdev:g}, lambda_vio={lam_vio_c:g}", flush=True)
    n_evs_c = int(cond[6]) if len(cond) > 6 else int(ev_cfg.n_evs)
    ev_cfg_c = ev_cfg if n_evs_c == ev_cfg.n_evs else EVCusConfig(n_evs=n_evs_c)
    algos = SCALING_ALGOS if SCALING else (T2_ALGOS if TABLE2 else ALGOS)
    hg_algos = tuple(a for a in algos if ALGO_SPECS[a]["hg"])
    prices, vpp_incen = tariff_setting(loca)
    T0 = time.time()
    TM = dict.fromkeys(_TIMING_KEYS, 0.0)
    SETUP = {"data_preprocessing": PREP_S}

    _use_seed_dir(seed)
    P = _paths(seed, cond_i)
    np.random.seed(int(seed)); _pyrandom.seed(int(seed)); _seed_torch(int(seed))
    _t = _pc()
    runner = create_vpu_runner(master_dss=MASTER_DSS_PATH)
    SETUP["opendss_setup"] = _pc() - _t
    _instrument_runner(runner, TM)
    _ensure_workdir()

    _t = _pc()
    agents, rngs = {}, {}
    for a in algos:
        sp = ALGO_SPECS[a]
        sp["cfg"]["seed"] = int(seed)
        _seed_torch(int(seed) + sp["init"])
        agents[a] = _disable_td3_diagnostics(MultiAgentSAC(sp["cfg"], float(buffer_frac) if sp["hg"] else 0.0))
        rngs[a] = _AlgoRNG(int(seed) * 100003 + sp["init"] + 17)
    SETUP["agent_init"] = _pc() - _t

    _t = _pc()
    envs = {a: _make_env(int(seed), vpp_incen, runner, BATT_DEG_WEIGHT, ev_cfg_c) for a in algos}
    for _e in envs.values():
        _e.disable_v2x = bool(NO_V2X and T2_NOV2X_MODE == "charge_only")
    train_fleet = generate_ev(ev_cfg_c, rng=np.random.default_rng([int(seed), 20240001]))
    if SCALING or n_evs_c != ev_cfg.n_evs:
        eval_fleet = generate_ev(ev_cfg_c, rng=np.random.default_rng([int(EVAL_SEED), 20240001]))
    else:
        eval_fleet = EVAL_FLEET.copy(deep=True)
    SETUP["env_generation"] = _pc() - _t

    acc = {a: _BehaviourAccumulator("train") for a in algos}
    acc_eval = {a: _BehaviourAccumulator("eval") for a in algos}
    day_reward = {a: np.zeros(DAYS) for a in algos}
    daily_rows, timing_rows, grid_rows = [], [], []
    soc_rows = {a: [] for a in algos}
    reserve_train = dict(zip(train_fleet["ev_id"], train_fleet["v2g_min_threshold"]))
    reserve_eval = dict(zip(eval_fleet["ev_id"], eval_fleet["v2g_min_threshold"]))
    batt_train = dict(zip(train_fleet["ev_id"], train_fleet["battery_kwh"]))
    batt_eval = dict(zip(eval_fleet["ev_id"], eval_fleet["battery_kwh"]))
    state = {a: None for a in algos}

    def consume_all():
        for a in algos:
            rows = envs[a].ablation_history
            if not rows:
                continue
            df = pd.DataFrame.from_records(rows)
            rows.clear()
            df.loc[df["day_idx"] < warmup, "policy"] = "Heuristic"
            acc[a].consume(df)
            acc_eval[a].consume(df, want_daily=False)
            if "participant" in df.columns:
                for st_name, part in df.groupby("stat"):
                    ev_ = st_name == "eval"
                    soc_rows[a] += soc_profile_rows(part, reserve_eval if ev_ else reserve_train,
                                                    batt_eval if ev_ else batt_train)

    for d_idx in range(DAYS):
        _td0, _cd0 = _pc(), _cpu()
        _tm0 = dict(TM)
        phase = "warmup" if d_idx < warmup else ("train" if d_idx < t_DAYS else "eval")
        stat = "eval" if phase == "eval" else "train"
        fleet = eval_fleet if phase == "eval" else train_fleet
        if phase == "eval":
            p_today = (float(participation_rate) if (TABLE2 and T2_FIXED_EVAL_P)
                       else float(EVAL_P_SCHEDULE[d_idx - t_DAYS]))
            part_rng = np.random.default_rng([int(EVAL_SEED), 123, d_idx])
        else:
            p_today = float(participation_rate)
            part_rng = np.random.default_rng(123 + d_idx)
        if NO_V2X and T2_NOV2X_MODE == "idle":
            participant_ids = set()                      
        else:
            participant_ids = sample_participating_evs(fleet, p_today, part_rng)
        vpp_req = bool(vpp_request[d_idx])
        active = hg_algos if phase == "warmup" else algos
        fresh_start = d_idx in (0, warmup, t_DAYS)

        if d_idx % 10 == 0 or d_idx in (warmup, t_DAYS, DAYS - 1):
            print(f"[progress] cond {cond_i}/{_N_COND_ALL} {tag} seed={seed} {phase.upper()} day {d_idx + 1}/{DAYS} "
                  f"| seed {(time.time() - T0) / 60:.1f} min | total {(time.time() - _T_RUN0) / 3600:.2f} h", flush=True)

        t0 = d_idx * STEPS_PER_DAY + peak_start
        fresh_travel = None
        _t, _d0 = _pc(), TM["opendss"]
        for a in active:
            env = envs[a]
            pol = "Heuristic" if phase == "warmup" else ALGO_SPECS[a]["policy"]
            st = state[a]
            if fresh_start or st is None:
                if fresh_travel is None:
                    env.dss_runner.reset_episode()
                    fresh_travel = env.v2g_ev_travel(fleet, d_idx, trip_cfg, stat)
                travel = fresh_travel
                vdev, vio = 0.0, 0.0
            else:
                travel = env.next_travel(d_idx, fleet, st["obs"], trip_cfg, stat)
                vdev, vio = st["vdev"], st["vio"]
            obs = {tr["ev_id"]: env.v2g_obs(t0, tr["ev_id"], fleet, travel, tr["soc_after_trip"], pol, "Idle",
                                            vpp_req, prices, stat, vdev, vio) for tr in travel}
            state[a] = {"obs": obs, "travel": travel, "vdev": vdev, "vio": vio, "policy": pol,
                        "env_policy": "Gaussian" if pol == "Heuristic" else pol,
                        "r": 0.0, "v2g": 0.0, "v2h": 0.0, "ch": 0.0, "rech": 0.0, "vdev_sum": 0.0, "vio_sum": 0.0}
        TM["trip_obs"] += (_pc() - _t) - (TM["opendss"] - _d0)

        for p_idx in range(peak_start, peak_end, 4):
            t_idx = d_idx * STEPS_PER_DAY + p_idx
            done = 1 if p_idx == (peak_end - 4) else 0
            for a in active:
                st, env, ag = state[a], envs[a], agents[a]
                pol, obs = st["policy"], st["obs"]
                _t = _pc()
                with rngs[a]:
                    if pol == "Heuristic":
                        act = warmup_explore_actions(obs, seed, d_idx, p_idx)
                    elif ALGO_SPECS[a]["td3"]:
                        act = ag.act(obs, explore=(phase == "train"), expl_noise=TD3_EXPL_NOISE)
                    else:
                        act = ag.act(obs, explore=(phase == "train"))
                act, obs = apply_participation_to_action(act, obs, participant_ids)
                TM["act"] += _pc() - _t
                _n_hist0 = len(env.ablation_history)
                _t, _d0 = _pc(), TM["opendss"]
                (nobs, rew, v2h, v2g, ch, rech, vdev, vio, _sens) = env.step(
                    t_idx, act, obs, fleet, st["env_policy"], st["travel"], vpp_req, a, p_idx, prices, stat,
                    participant_ids, vdev_max, vdev_min, lambda_vdev, lam_vio_c)
                TM["env_step"] += (_pc() - _t) - (TM["opendss"] - _d0)
                _t = _pc()
                if TABLE2:
                    grid_rows.append({"day_idx": d_idx, "p_idx": p_idx, "t_idx": t_idx, "phase": phase,
                                      "stat": stat, "algo": a, "vpp_req": bool(vpp_req),
                                      "n_participants": len(participant_ids), **_grid_step_row(env, vdev, vio)})
                for _row in env.ablation_history[_n_hist0:]:
                    _aid = _row["agent_id"]
                    _row["participant"] = _aid in participant_ids
                    _row["connected"] = bool(obs[_aid].get("connectivity", False)) if _aid in obs else False
                TM["logging"] += _pc() - _t
                _t = _pc()
                if phase != "eval":
                    ag.observe(act, obs, nobs, rew, pol, done)
                TM["buffer"] += _pc() - _t
                _t = _pc()
                if phase == "train":
                    with rngs[a]:
                        ag.learn()
                TM["nn_update"] += _pc() - _t
                st["r"] += _ev_sum(rew)
                st["v2g"] += _ev_sum(v2g); st["v2h"] += _ev_sum(v2h)
                st["ch"] += _ev_sum(ch); st["rech"] += _ev_sum(rech)
                st["vdev_sum"] += float(vdev); st["vio_sum"] += float(vio)
                st["obs"], st["vdev"], st["vio"] = nobs, float(vdev), float(vio)

        _t = _pc()
        n_steps = len(range(peak_start, peak_end, 4))
        for a in active:
            st = state[a]
            r_day = st["r"] / n_evs_c
            if phase != "warmup":
                day_reward[a][d_idx] = r_day
            daily_rows.append({
                "day_idx": d_idx, "phase": phase, "policy": st["policy"], "algo": a,
                "algorithm": ALGO_SPECS[a]["name"], "vpp_request": bool(vpp_req),
                "participation_rate_day": p_today, "n_participants": len(participant_ids),
                "reward": r_day, "V2G_kWh": st["v2g"], "V2H_kWh": st["v2h"], "charging_kWh": st["ch"],
                "recharge_kWh": st["rech"], "voltage_deviation": st["vdev_sum"] / n_steps,
                "violation_rate": st["vio_sum"] / n_steps,
            })
        consume_all()
        TM["logging"] += _pc() - _t
        if SCALING:
            _wall = _pc() - _td0
            _row = {"day_idx": d_idx, "phase": phase, "wall_total_s": _wall, "cpu_total_s": _cpu() - _cd0}
            for k in _TIMING_KEYS:
                _row[f"{k}_s"] = TM[k] - _tm0[k]
            _row["other_s"] = _wall - sum(_row[f"{k}_s"] for k in _TIMING_KEYS)
            timing_rows.append(_row)

    _fn = lambda pfx: (f"{pfx}_reward_{tag}_{n_evs_c}_{seed}_{participation_rate}_{loca}_{reward_w}_"
                       f"{lambda_vdev}_{lam_vio_c}_{buffer_frac}.txt")
    for a in algos:
        pfx = ALGO_SPECS[a]["prefix"]
        np.savetxt(_fn(pfx), day_reward[a][:t_DAYS], fmt='%.5f', delimiter=',')
        np.savetxt("eval_" + _fn(pfx), day_reward[a][t_DAYS:], fmt='%.5f', delimiter=',')

    _ensure_workdir()
    consume_all()
    meta = {"sweep_type": SWEEP_TYPE, "condition": tag, "deg_weight": float(BATT_DEG_WEIGHT),
            "vdev_weight": float(lambda_vdev), "lambda_vio": float(lam_vio_c), "participation_rate": float(participation_rate),
            "rho": float(buffer_frac), "tariff": loca, "seed": int(seed), "td3_noise": float(TD3_EXPL_NOISE),
            "eval_seed": int(EVAL_SEED), "n_evs": n_evs_c}
    _append_csv(P["summary"], pd.DataFrame([acc[a].summary_row({**meta, "algo": a}) for a in algos]))
    _append_csv(P["eval"], pd.DataFrame([acc_eval[a].summary_row({**meta, "algo": a}) for a in algos]))
    _append_csv(P["soc"], pd.concat([acc[a].soc_df({**meta, "algo": a}) for a in algos], ignore_index=True))
    _append_csv(P["daily"], pd.concat([acc[a].daily_df({**meta, "algo": a}) for a in algos], ignore_index=True))
    ddf = pd.DataFrame(daily_rows)
    for k, v in meta.items():
        ddf[k] = v
    _append_csv(P["daily_reward"], ddf)
    _sp = []
    for a in algos:
        q = pd.DataFrame(soc_rows[a])
        if q.empty:
            continue
        q["phase"] = np.where(q["stat"] == "eval", "eval", np.where(q["day_idx"] < warmup, "warmup", "train"))
        q["algo"], q["algorithm"] = a, ALGO_SPECS[a]["name"]
        _sp.append(q)
    if _sp:
        spdf = pd.concat(_sp, ignore_index=True)
        for k, v in meta.items():
            spdf[k] = v
        _append_csv(P["soc_profile"], spdf)
    if TABLE2 and grid_rows:
        gdf = pd.DataFrame(grid_rows)
        for k, v in meta.items():
            gdf[k] = v
        gdf["lambda_vio"] = float(lam_vio_c)
        _append_csv(P["grid_steps"], gdf)
        summ = (gdf[gdf["phase"] != "warmup"]
                .groupby(["algo", "phase"], sort=False)
                .agg(n_steps=("t_idx", "size"),
                     n_violation_steps=("step_violation", "sum"),
                     n_bus_violations=("n_bus_violation", "sum"),
                     mean_voltage_deviation=("voltage_deviation", "mean"),
                     mean_violation_rate=("violation_rate", "mean"),
                     min_vpu=("vpu_min", "min"), max_abs_dev=("max_abs_dev", "max"),
                     mean_participants=("n_participants", "mean"))
                .reset_index())
        for k, v in meta.items():
            summ[k] = v
        summ["lambda_vio"] = float(lam_vio_c)
        _append_csv(P["grid"], summ)
    if SCALING and timing_rows:
        tdf = pd.DataFrame([{"day_idx": -1, "phase": "setup", **{f"{k}_s": v for k, v in SETUP.items()},
                             "wall_total_s": float(sum(SETUP.values()))}] + timing_rows)
        tdf["algo"] = ",".join(algos)
        for k, v in meta.items():
            tdf[k] = v
        _append_csv(P["timing"], tdf)

    for a in algos:
        envs[a].ablation_history.clear(); envs[a]._home_load_hist.clear()
    del agents, envs, runner, acc, acc_eval
    gc.collect()
    if _CUDA:
        torch.cuda.empty_cache()


_CSV_PATTERNS = ("all_reward_weight_sensitivity_seed_summary", "all_eval_summary", "all_soc_trajectory",
                 "all_daily_action_counts", "all_daily_rewards", "all_soc_profile")


def expand_aliases(aliases):
    n_txt = n_csv = 0
    for s in seeds:
        sd = OUT_ROOT / f"seed{s}"
        if not sd.is_dir():
            continue
        for ai, oi in aliases:
            alias, t_a, t_o = ALL_STUDIES[ai - 1], cond_tag(ALL_STUDIES[ai - 1]), cond_tag(ALL_STUDIES[oi - 1])
            for f in sd.glob(f"*_reward_{t_o}_*.txt"):
                shutil.copy(f, sd / f.name.replace(f"_reward_{t_o}_", f"_reward_{t_a}_"))
                n_txt += 1
            for pat in _CSV_PATTERNS:
                f = sd / f"{pat}{_stag(s, oi)}.csv"          
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                q = df[df["condition"] == t_o].copy()
                if q.empty:
                    continue
                q["condition"], q["sweep_type"] = t_a, alias[0]
                q.to_csv(sd / f"{pat}{_stag(s, ai)}.csv", index=False)   
                n_csv += 1
    print(f"[alias] {len(aliases)} conditions copied: {n_txt} txt, {n_csv} csv", flush=True)


import shutil
for _cond_i, cond in _RUN_LIST:
    tag = cond_tag(cond)
    for _seed in [s for s in seeds if cond[0] != "scaling" or s in SCALING_SEEDS]:
        _use_seed_dir(_seed)
        _done_marker = Path(f"_done_{tag}{_stag(_seed, _cond_i)}")
        if RESUME and _done_marker.exists():
            continue
        run_one_seed(int(_seed), _cond_i, cond, tag)
        _use_seed_dir(_seed)
        _done_marker.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")

if _ALIASES:
    expand_aliases(_ALIASES)
_CUR_DIR = OUT_ROOT
_ensure_workdir()
