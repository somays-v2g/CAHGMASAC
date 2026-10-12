from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

      
@dataclass
class EVCusConfig:
    def __init__(
        self,
        n_evs: int = 250,
    ):
        
        self.n_evs = n_evs
        
        self.battery_kwh_mean: float = 70.0
        self.battery_kwh_std: float = 5.0
        self.battery_kwh_min: float = 20.0
        self.battery_kwh_max: float = 100.0
        self.char_soc_min: float = 0.40
        self.char_soc_min_std: float = 0.05
        self.v2g_soc_min: float = 0.60
        self.v2g_soc_min_std: float = 0.05
        self.init_soc_max: float = 1.00
        self.target_soc: float = 1.00
        self.min_soc_reserve: float = 0.40
        self.min_soc_reserve_std: float = 0.05

        self.charging_lock = False       
        self.discharging_lock = False    

        self.home_type_spec: Dict[str, Tuple[float, float]] = {
                "A": (2.4, 0.6),
                "B": (1.8, 0.4),
                "C": (1.2, 0.2),
        }

def generate_ev(
    cfg: EVCusConfig,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:

    N = cfg.n_evs
    if rng is None:
        rng = np.random.default_rng(seed)

    home_type = []
    average_kw = []
    std_kw = []

    batt = rng.normal(cfg.battery_kwh_mean, cfg.battery_kwh_std, cfg.n_evs) 
    batt = np.clip(batt, cfg.battery_kwh_min, cfg.battery_kwh_max)

    charge_threshold = rng.normal(cfg.char_soc_min,
                                  cfg.char_soc_min_std, cfg.n_evs).round(1)
    v2g_threshold = rng.normal(cfg.v2g_soc_min,
                               cfg.v2g_soc_min_std, cfg.n_evs).round(1)
    v2g_min_threshold = rng.normal(cfg.min_soc_reserve,
                                   cfg.min_soc_reserve_std, cfg.n_evs).round(1)

    for i in range(N):
        type = rng.choice(["A", "B", "C"])
        home_type.append(type)
        avg, std = cfg.home_type_spec[type]
        average_kw.append(avg)
        std_kw.append(std)


    init_soc = rng.normal(0.80, 0.10, cfg.n_evs)
    init_soc = np.clip(init_soc, 0.50, 1.00).round(2)


    df = pd.DataFrame({
        "ev_id": [f"EV_{i+1:03d}" for i in range(cfg.n_evs)],
        "battery_kwh": batt,
        "soc_start": init_soc,
        "charge_threshold": charge_threshold,
        "v2g_threshold": v2g_threshold,
        "v2g_min_threshold": v2g_min_threshold,
        "home_type": home_type,
        "average_kw": average_kw,
        "std_kw": std_kw,
    })

    df["soc_start"] = df["soc_start"].clip(0.0, 1.0)
    df["mode"] = "idle"  
        
    
    return df
