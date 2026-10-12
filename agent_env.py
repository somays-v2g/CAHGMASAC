
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Sequence
import csv

from gen_ev_model import EVCusConfig, generate_ev
from investigate_Vpu import create_vpu_runner

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import sys, os


sys.path.append(".") 
csv_path = r"./csv_inputs/v2ghome_bus.csv"  
price_path = r"./csv_inputs/price_t.csv"
DEF_V2G_DIR_PATH = r"./sac/def_output"
SAC_V2G_DIR_PATH = r"./sac/v2g_output"
SAC_V2H_DIR_PATH = r"./sac/v2h_output"
TD3_V2G_DIR_PATH = r"./td3/v2g_output"
TD3_V2H_DIR_PATH = r"./td3/v2h_output"
SAC_DEF_DIR_PATH = r"./dsac/v2g_output"
TD3_DEF_DIR_PATH = r"./dtd3/v2h_output"
 

def load_prices_from_csv(path: str) -> np.ndarray:
    
    prices = pd.read_csv(path, header=None, index_col=None).to_numpy(dtype=np.float32)
    
    return prices

@dataclass
class V2GParams:
    eta_discharge: float = 0.95   
    eta_charge: float = 0.95      
    cha_cap_kw: float = 7.0       
    v2g_cap_kw: float = 7.0       
    idle_recharge_key: str = "recharge_needed"  
    home_load_key: str = "home_load"            


STEPS = 96   
DAY_END_MIN = 60 * 24

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def time_to_minutes(hour, minute):
    return hour * 60 + minute

def decode_tidx(t_idx, STEPS=96):
    day = t_idx // STEPS + 1        
    step = t_idx % STEPS            
    hour = (step * 15) // 60
    minute = (step * 15) % 60
    return day, step, hour, minute

def encode_day_min_to_tidx(day, minute, STEPS=96, min_per_step=15):
    step_of_day = minute // min_per_step
    return (day - 1) * STEPS + step_of_day

def load_ev_to_bus_map(csv_path):
    V2G_to_bus = {}
    V2G_to_file = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            row = [col.strip() for col in row]
            
            if len(row) < 3:
                continue

            ev_name  = row[0]  # EV_001
            file_name = row[1]
            bus_name = row[2]  # 1.1
            V2G_to_bus[ev_name] = bus_name
            V2G_to_file[ev_name] = file_name

    return V2G_to_bus, V2G_to_file

peak_s = 16
peak_e = 21

def disc_zone(t_idx):
    day, step, hour, minute = decode_tidx(t_idx)
    if hour >= 0 and hour < 6:
        discount_zone = True
    else:
        discount_zone = True

    return discount_zone

V2G_to_bus, V2G_to_file = load_ev_to_bus_map(csv_path)

@dataclass
class TripConfig:
    dist_min_miles: float = 20.0
    dist_max_miles: float = 45.0
    start_time_range_min: int = time_to_minutes(7, 00)
    start_time_range_max: int = time_to_minutes(8, 00)
    arrive_time_range_min: int = time_to_minutes(16, 00)
    arrive_time_range_max: int = time_to_minutes(20, 00)
    kwh_per_mile_mean: float = 0.3583
    kwh_per_mile_std: float = 0.0715
    kwh_per_mile_min: float = 0.25
    kwh_per_mile_max: float = 0.53

@dataclass
class ChargerConfig:
    cha_rate_kw: float = 7.0
    dis_rate_kw: float = 7.0
    chg_efficiency: float = 0.95
    dis_efficiency: float = 0.95
    efficiency_std: float = 0.02

@dataclass
class V2GIncen:
    V2G_Failure_Panelty: float = 10.0

@dataclass
class DermsParams:
    C1: float = 10    
    C2: float = 100   
    C3: float = 100   
    C4: float = 10  
    C5: float = 5  
    vpp_threshold: float = 0.952
    vpp_threshold_up: float = 1.04
    critical_Vpu_down: float = 0.95
    critical_Vpu_up: float = 1.05


class V2GHomeEnv:
    def __init__(
        self,
        step_day: int = STEPS,
        peak_start: int = peak_s,
        peak_end: int = peak_e,
        trip_cfg: TripConfig | None = None,
        chg_cfg: ChargerConfig | None = None,
        ev_cfg: EVCusConfig | None = None,
        incen_cfg: V2GIncen | None = None,
        derms_cfg: DermsParams | None = None,
        seed_list: Sequence[int] | None = None, #int = 123,
        ev_seed: None = None,
        seed_train: Sequence[int] = (1,),
        VPP_incentive: int = 2,
        re_w: int = 10,
        dss_runner = None,
        seed_eval: int = 123,
    ):
        self.step_day = step_day
        self.peak_start = peak_start
        self.peak_end = peak_end
        
        self.dss_runner = dss_runner if dss_runner is not None else create_vpu_runner()
        
        self.trip_cfg = trip_cfg or TripConfig()
        self.chg_cfg = chg_cfg or ChargerConfig()
        self.ev_cfg = ev_cfg or EVCusConfig()
        self.VPP_incentive = VPP_incentive
        self.re_w = re_w

        self.use_batt_deg = True
        self.batt_deg_weight = 1.0           
        self.freeze_r_max = True
        self.grid_penalty_share = "per_agent"
        self.grid_penalty_scale = 0.1
        self.shared_exogenous = True
        self.resample_fleet_each_reset = True

        self.seed_train = seed_train
        self.seed_t = int(seed_train[0])      
        self.seed_e = int(seed_eval)
        self.train_rng = np.random.default_rng(self.seed_t)
        self.eval_rng = np.random.default_rng(self.seed_e)
        self.fleet_rng = np.random.default_rng([self.seed_t, 20240001])
        self._fleet_cache = None
        self.incen_cfg = incen_cfg or V2GIncen()
        self.derms_cfg = derms_cfg or DermsParams()
        self._buf = {"hsac": {}, "sac": {}, "htd3": {}, "td3": {}, "dsac": {}, "dtd3": {}}
        self._home_load_hist = {}
        self._vpp_req_hist = {}
        self.ablation_history = []

        self._ev_lookup_cache_key = None
        self._ev_cache_refs = (None, None)
        self._ev_initial_map = {}
        self._ev_travel_map = {}
        self.keep_power_buffer = False

    def _ensure_ev_lookup_cache(self, ev_initial, ev_travel):
        key = (id(ev_initial), id(ev_travel))
        ref_i, ref_t = self._ev_cache_refs
        if (ref_i is not ev_initial) or (ref_t is not ev_travel) or key != self._ev_lookup_cache_key:
            self._ev_cache_refs = (ev_initial, ev_travel)
            if hasattr(ev_initial, "set_index"):
                self._ev_initial_map = ev_initial.set_index("ev_id", drop=False).to_dict(orient="index")
            else:
                self._ev_initial_map = {str(row["ev_id"]): row for row in ev_initial}
            self._ev_travel_map = {str(row["ev_id"]): row for row in ev_travel}
            self._ev_lookup_cache_key = key
        return self._ev_initial_map, self._ev_travel_map


    @staticmethod
    def _ev_index(ev_id) -> int:
        try:
            return int(str(ev_id).split("_")[-1])
        except Exception:
            return abs(hash(str(ev_id))) % 1_000_000

    _EVAL_SALTS = (102, 202, 212)

    def _exo_rng(self, salt: int, *keys) -> np.random.Generator:
        base = self.seed_e if int(salt) in self._EVAL_SALTS else self.seed_t
        return np.random.default_rng(
            [int(base), int(salt)] + [int(k) for k in keys]
        )

    def sample_eff(self, eff_mean, eff_std, stat):
        if stat == 'train':
            eff = float(self.train_rng.normal(eff_mean, eff_std))
        elif stat == 'eval':
            eff = float(self.eval_rng.normal(eff_mean, eff_std))
        else:
            print("stat is worng")
        
        return eff

    def sample_trip_params(self, n: int, tcfg: TripConfig, rng: np.random.Generator):
        miles = rng.uniform(tcfg.dist_min_miles, tcfg.dist_max_miles, n)
        start = int(np.asarray(rng.integers(tcfg.start_time_range_min, tcfg.start_time_range_max+1, n)).reshape(-1)[0])
        arrive = rng.integers(tcfg.arrive_time_range_min, tcfg.arrive_time_range_max+1, n)
        arrive = np.where(arrive <= start, start + np.maximum(30, rng.integers(60, 240, n)), arrive)
        arrive = int(np.asarray(np.clip(arrive, 0, DAY_END_MIN)).reshape(-1)[0])
        kpm = rng.normal(tcfg.kwh_per_mile_mean, tcfg.kwh_per_mile_std, n)
        kpm = np.clip(kpm, tcfg.kwh_per_mile_min, tcfg.kwh_per_mile_max)
        if n == 1:
            return float(miles[0]), start, arrive, float(kpm[0])
        return miles, start, arrive, kpm
    
   
    def v2g_ev_travel(
            self,
            ev_travel,
            day_idx: int,
            tcfg: TripConfig,
            stat,
    ) -> Dict[str, Any]:
    
        after_travel: List[Dict[str, Any]] = []

        for num in range(len(ev_travel)):
            ev_id = ev_travel.iloc[num]["ev_id"]
            batt_cap = ev_travel.iloc[num]["battery_kwh"]
            initial_soc = ev_travel.iloc[num]["soc_start"]

            if self.shared_exogenous:
                _r = self._exo_rng(201 if stat == 'train' else 202,
                                   day_idx, self._ev_index(ev_id))
                miles, start, arrive, kpm = self.sample_trip_params(1, tcfg, _r)
            elif stat == 'train':
                miles, start, arrive, kpm = self.sample_trip_params(1, tcfg, self.train_rng)
            elif stat == 'eval':
                miles, start, arrive, kpm = self.sample_trip_params(1, tcfg, self.eval_rng)
            else:
                print("stat is wrong")
            energy_drive = miles * kpm  # kWh

            use_kwh = float(energy_drive)
            d_soc = use_kwh / batt_cap
            soc_after_trip = clamp(initial_soc - d_soc, 0.0, 1.0)
     
            after_travel.append({
                "day_idx": day_idx,
                "ev_id": ev_id,
                "arrival": arrive,
                "next_start": start,
                "soc_after_trip": soc_after_trip,
            })

        return after_travel
    
    def next_travel(
            self,
            day_idx: int,
            ev_initial,
            next_obs_by_agent,
            trip_cfg: TripConfig,
            stat,
    ) -> Dict[str, Any]:
    
        nextday_travel: List[Dict[str, Any]] = []

        for num in range(len(ev_initial)):
            ev_id = ev_initial.iloc[num]["ev_id"]
            batt_cap = ev_initial.iloc[num]["battery_kwh"]
            current_soc = next_obs_by_agent[ev_id]["current_soc"]

            if self.shared_exogenous:
                _r = self._exo_rng(211 if stat == 'train' else 212,
                                   day_idx, self._ev_index(ev_id))
                miles, start, arrive, kpm = self.sample_trip_params(1, trip_cfg, _r)
            elif stat == 'train':
                miles, start, arrive, kpm = self.sample_trip_params(1, trip_cfg, self.train_rng)
            elif stat == 'eval':
                miles, start, arrive, kpm = self.sample_trip_params(1, trip_cfg, self.eval_rng)
            else:
                print("stat is wrong")
            energy_drive = miles * kpm  # kWh

            use_kwh = float(energy_drive)
            d_soc = use_kwh / batt_cap
            soc_after_trip = clamp(current_soc - d_soc, 0.0, 1.0)
     
            nextday_travel.append({
                "day_idx": day_idx,
                "ev_id": ev_id,
                "arrival": arrive,
                "next_start": start,
                "soc_after_trip": soc_after_trip,
            })

        return nextday_travel
    
    def eval_ev_travel(
            self,
            ev_initial,
            day_idx: int,
            tcfg: TripConfig,
            stat,
            ev_cfg,
    ) -> Dict[str, Any]:
    
        eval_init_travel: List[Dict[str, Any]] = []

        for num in range(len(ev_initial)):
            ev_id = ev_initial.iloc[num]["ev_id"]
            batt_cap = ev_initial.iloc[num]["battery_kwh"]
            init_soc = self.eval_rng.normal(0.80, 0.10, 1)
            initial_soc = float(np.clip(init_soc, 0.50, 1.00).round(2)[0])

            if stat == 'train':
                miles, start, arrive, kpm = self.sample_trip_params(1, tcfg, self.train_rng)
            elif stat == 'eval':
                miles, start, arrive, kpm = self.sample_trip_params(1, tcfg, self.eval_rng)
            else:
                print("stat is wrong")
            energy_drive = miles * kpm  # kWh

            use_kwh = float(energy_drive)
            d_soc = use_kwh / batt_cap
            soc_after_trip = clamp(initial_soc - d_soc, 0.0, 1.0)
     
            eval_init_travel.append({
                "day_idx": day_idx,
                "ev_id": ev_id,
                "arrival": arrive,
                "next_start": start,
                "soc_after_trip": soc_after_trip,
            })

        return eval_init_travel
    
    
    def v2g_obs(
            self,
            t_idx: int,
            evid,
            ev_initial, 
            ev_travel,
            current_soc,
            policy: str,
            current_act,
            vpp_req: bool,
            prices,
            stat,
            voltage_deviation,
            violation_rate,
    ) -> Dict[str, Any]:
        
        v2g_obs: List[Dict[str, Any]] = []

        day, step, hour, minute = decode_tidx(t_idx)

        ev_initial_map, ev_travel_map = self._ensure_ev_lookup_cache(ev_initial, ev_travel)
        ev_key = str(evid)
        init_ev = ev_initial_map[ev_key]
        travel_ev = ev_travel_map[ev_key]
        charge_thres = init_ev["charge_threshold"]
        v2g_allow_soc = init_ev["v2g_threshold"]
        V2G_min_thres = init_ev["v2g_min_threshold"]
        home_kw_mean = init_ev["average_kw"]
        home_kw_std = init_ev["std_kw"]

        if self.shared_exogenous:
            _r = self._exo_rng(101 if stat == 'train' else 102,
                               t_idx, self._ev_index(evid))
            home_energy = float(np.clip(_r.normal(home_kw_mean, home_kw_std), 0.0, None))
        elif stat == 'train':
            home_energy = float(np.clip(self.train_rng.normal(home_kw_mean, home_kw_std), 0.0, None))
        elif stat == 'eval':
            home_energy = float(np.clip(self.eval_rng.normal(home_kw_mean, home_kw_std), 0.0, None))
        else:
            print("stat is wrong")


        arr_hour = travel_ev['arrival'] // 60
        sta_hour = travel_ev['next_start'] // 60

        if (hour >= arr_hour) and (hour <= 23):
            connectivity = True
        elif (hour >= 0) and (hour < sta_hour):
            connectivity = True
        else:
            connectivity = False

        soc_after_trip = travel_ev['soc_after_trip']
        arrival = arr_hour
        
        if soc_after_trip >= v2g_allow_soc and arrival < peak_e:
            V2G_availability = 1
        else:
            V2G_availability = 0

        V2H_power = home_energy
        V2G_power = self.chg_cfg.dis_rate_kw - home_energy

        v2g_obs = {
            "t_idx": t_idx,
            "ev_id": evid,
            "connectivity": connectivity,
            "policy": policy,
            "current_act": current_act,
            "current_soc": current_soc,
            "vpp_req": vpp_req,
            "soc_after_trip": soc_after_trip,
            "time_to_arrival": arrival,
            "V2G_availability": V2G_availability,
            "V2H_power": V2H_power,
            "V2G_power": V2G_power,
            "price": float(prices[t_idx]),
            "home_energy": home_energy,
            "charge_thres": charge_thres,
            "V2G_min_thres": V2G_min_thres,
            "voltage_deviation": voltage_deviation,
            "violation_rate": violation_rate,
        }

        return v2g_obs
    
    def derms_obs_vpp(
            self,
            vpu_data,
            t_idx: int,
            policy: str,
            vpp_req_pre: bool
    ) -> Dict[str, Any]:


        day, step, hour, minute = decode_tidx(t_idx)
        vpu_row = vpu_data["vpu"].dropna()
        vpu_values = vpu_row.values.astype(float)
        bus_ids = list(vpu_row.index)
        #print(vpu_row)
        #print(vpu_values)
        #print(bus_ids)

        vpu_min = float(vpu_values.min())
        vpu_max = float(vpu_values.max())
        vpu_mean = float(vpu_values.mean())

        summary = {
            "vpu_min": vpu_min,
            "vpu_max": vpu_max,
            "vpu_mean": vpu_mean,
            "num_buses": len(bus_ids),
        }

        derms_obs = {
            "t_idx": t_idx,
            "vpp_req": vpp_req_pre,
            "summary": summary
        }

        return derms_obs

    def build_multiagent_obs(
        self,
        v2g_home_obs_pre: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        
        obs_by_agent = v2g_home_obs_pre
        
        return obs_by_agent
       
    
    def reset(self, day_idx, time_index, v2g_ap, stat, trip_cfg):
        
        self.dss_runner.reset_episode()
        
        if self.resample_fleet_each_reset or self._fleet_cache is None:
            self._fleet_cache = generate_ev(self.ev_cfg, rng=self.fleet_rng)
        ev_initial = self._fleet_cache.copy(deep=True)
        ev_travel = self.v2g_ev_travel(ev_initial, day_idx, trip_cfg, stat)
        self._ev_lookup_cache_key = None
        self._ensure_ev_lookup_cache(ev_initial, ev_travel)

        T = time_index  
        ev_ids = [e["ev_id"] for e in ev_travel]

        if self.keep_power_buffer:
            for algo in ("hsac", "sac", "htd3", "td3", "dsac", "dtd3"):
                for ev_id in ev_ids:
                    self._buf[algo][ev_id] = np.zeros(T, dtype=np.float32)

        return ev_initial, ev_travel
 
    def next_state(self, t_idx, real_action_by_agent, policy, ev_travel, vpp_req, ev_initial, prices, stat, voltage_deviation, violation_rate):


        v2g_home_obs = {}
        next_obs_by_agent: List[Dict[str, Any]] = []


        for n in range(len(ev_travel)):
            ev_id = ev_travel[n]['ev_id']
            current_soc = real_action_by_agent[ev_id]['current_soc']
            current_act = real_action_by_agent[ev_id]['next_action']

            v2g_obs = self.v2g_obs(t_idx, ev_id, ev_initial, ev_travel, current_soc, policy, current_act, vpp_req, prices, stat, voltage_deviation, violation_rate)
            v2g_home_obs[ev_id] = v2g_obs
        
        next_obs_by_agent = v2g_home_obs

        return next_obs_by_agent

    def home_v2g_reward(self,
                        next_action: str,
                        V2G_amount: float,
                        V2H_amount: float,
                        CH_amount: float,
                        prices_t: float,
                        incen_cfg: V2GIncen,
                        panelty: int,
                        ev_id: str | None = None,
                        time_slot: int | None = None,
                        _home_load_hist: float | None = None,
                        vpp_req: bool | None = False,
                        policy: str | None = "Gaussian",
                    ) -> float:
        
        reward_home = 0.0
        
        if policy == "Default":
            
            if next_action == "V2G":
                if vpp_req == True:
                    reward_home += (V2G_amount * self.VPP_incentive)
                else:
                    reward_home -= (V2G_amount * prices_t)
            elif next_action == "V2H":
                reward_home = 0
            elif next_action == "Charging":
                
                if vpp_req == True:
                    reward_home -= (CH_amount * prices_t) + (V2H_amount * prices_t) + ((7 - V2H_amount) * self.VPP_incentive)
                else:
                    reward_home -= (CH_amount * prices_t) + (V2H_amount * prices_t)   
            else:
                reward_home -= (V2H_amount * prices_t)
            
        else:
            if next_action == "V2G":
                #reward_home += (V2H_amount * prices_t) 
                reward_home += (V2G_amount * self.VPP_incentive)
            elif next_action == "V2H":
                pass  
            elif next_action == "Charging":
                if vpp_req == True: #panelty == 2:
                    reward_home -= (CH_amount * prices_t) + (V2H_amount * prices_t) + ((7 - V2H_amount) * self.VPP_incentive)
                else:
                    reward_home -= (CH_amount * prices_t) + (V2H_amount * prices_t)
            else:
                if panelty == 2:
                    reward_home -= (V2H_amount * prices_t)   
                else:
                    reward_home -= (V2H_amount * prices_t)   
        return reward_home

    def derms_reward(self, obs_by_agent: Dict[str, Any], derms_cfg: DermsParams) -> float:
        v = float(obs_by_agent['summary']['vpu_min'])
        vpp_req = obs_by_agent['vpp_req']
        if v <= derms_cfg.critical_Vpu_down:
            derms_reward_t = -derms_cfg.C2 * (0.95 - v)
        elif derms_cfg.critical_Vpu_down < v <= derms_cfg.vpp_threshold: 
            derms_reward_t = -derms_cfg.C1 * (0.96 - v)
        elif v >= derms_cfg.critical_Vpu_up:
            derms_reward_t = -derms_cfg.C3 / (v - 1.05)
        elif derms_cfg.critical_Vpu_up > v >= derms_cfg.vpp_threshold_up: 
            derms_reward_t = -derms_cfg.C4 * (v - 1.04)
        elif (derms_cfg.vpp_threshold_up > v > derms_cfg.vpp_threshold) and (vpp_req == True):
            derms_reward_t = derms_cfg.C2 / 100
        else:
            derms_reward_t = derms_cfg.C5
        return derms_reward_t

    def compute_reward(self,
                       aid,
                       reward_derms,
                       reward_home,
                       ) -> Dict[str, float] | float:
     
        reward_by_agent = {}
    
        reward_by_agent["DERMS"] = reward_derms
        reward_by_agent[aid] = reward_home
    
        return reward_by_agent

    def add_power_to_buffer(self, ev_id, t_idx, power_sum, next_action, policy, algo):
        if next_action in ("V2G", "V2H"):
            load = power_sum / 7
        elif next_action == "Charging":
            load = -power_sum / 7
        else:
            return

        if algo in ("hsac", "sac", "htd3", "td3", "dsac", "dtd3"):
            self._buf[algo][ev_id][t_idx] += load
            self._buf[algo][ev_id][t_idx+1] += load
            self._buf[algo][ev_id][t_idx+2] += load
            self._buf[algo][ev_id][t_idx+3] += load

                
    def flush_buffers_to_csv_4types(self, ev_travel, V2G_to_file, algo, policy):
    
        out_dir = {
            "hsac": SAC_V2G_DIR_PATH, "sac": SAC_V2H_DIR_PATH,
            "htd3": TD3_V2G_DIR_PATH, "td3": TD3_V2H_DIR_PATH,
            "dsac": SAC_DEF_DIR_PATH, "dtd3": TD3_DEF_DIR_PATH
        }
        
        if algo in ("hsac", "sac", "htd3", "td3", "dsac", "dtd3"):
            for ev in ev_travel:
                #print(algo, ev)
                ev_id = ev["ev_id"]
                file_name = V2G_to_file[ev_id]

                dir_path = out_dir[algo]
                file_path = os.path.join(dir_path, f"{file_name}.csv")
            
                pd.DataFrame(self._buf[algo][ev_id]).to_csv(
                    file_path, index=False, header=False
                )



    def step(self, t_idx, action_by_agent, obs_by_agent, ev_initial, policy, ev_travel, vpp_req, algo, p_idx, prices, stat, participant_ids, vdev_max, vdev_min, lambda_vdev, lambda_vio):        
        

        day, step, hour, minute = decode_tidx(t_idx)
        reward_by_agent = {}
        batt_deg = {}
        real_action_by_agent = {}
        next_obs_by_agent = {}
        V2H_by_agent = {}
        V2G_by_agent = {}
        CH_by_agent = {}        
        RECH_by_agent = {}
        sensitivity = {}
        step_diagnostics = {}
        storage_multiplier = {}
        temp_reward = []

        ev_initial_map, ev_travel_map = self._ensure_ev_lookup_cache(ev_initial, ev_travel)

        for aid, action in action_by_agent.items():
            reward_by_agent[aid] = reward_by_agent.get(aid, 0)
            batt_deg[aid] = batt_deg.get(aid, 0)
            V2H_by_agent[aid] = V2H_by_agent.get(aid, 0)
            V2G_by_agent[aid] = V2G_by_agent.get(aid, 0)
            CH_by_agent[aid] = CH_by_agent.get(aid, 0)
            RECH_by_agent[aid] = RECH_by_agent.get(aid, 0)
            real_action_by_agent[aid] = real_action_by_agent.get(aid, 0)
            reward_home = 0
            panelty = 0
            
            if aid != "DERMS":
                obs_ev = obs_by_agent[aid] 
                peak = obs_ev.get('is_peak', True) 
                connectivity = obs_ev.get('connectivity', True)
                pre_action = obs_ev.get('current_act', 'Idle')
                current_soc = obs_ev.get('current_soc', 1.0)
                ev_key = str(aid)
                travel_ev = ev_travel_map[ev_key]
                init_ev = ev_initial_map[ev_key]
                soc_after_trip = travel_ev['soc_after_trip']
                next_start = travel_ev['next_start']
                time_to_arrival = travel_ev['arrival']
            
                battery_kwh = init_ev["battery_kwh"]
                soc_start = init_ev["soc_start"]
                charge_threshold = init_ev["charge_threshold"]
                v2g_threshold = init_ev["v2g_threshold"]
                v2g_min_threshold = init_ev["v2g_min_threshold"]
                target_soc = 1.0
                average_kw = init_ev["average_kw"]
                std_kw = init_ev["std_kw"]

                if "home_energy" in obs_ev:
                    home_power = max(0.0, float(obs_ev["home_energy"]))
                elif stat == 'train':
                    home_power = float(np.clip(self.train_rng.normal(average_kw, std_kw), 0.0, None))
                elif stat == 'eval':
                    home_power = float(np.clip(self.eval_rng.normal(average_kw, std_kw), 0.0, None))
                else:
                    raise ValueError(f"Unknown stat: {stat!r}")
                    

                charging_power = self.chg_cfg.cha_rate_kw * self.chg_cfg.chg_efficiency
                discharging_power = self.chg_cfg.dis_rate_kw * self.chg_cfg.dis_efficiency
                if discharging_power > 7:
                    print("error")
                
                reward_home = 0
                re_CH_reward_home = 0

                if connectivity == True:
                    if (policy == "Heuristic"):
                        if (soc_after_trip >= v2g_threshold) and (current_soc > v2g_min_threshold):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 1
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        elif (soc_after_trip >= v2g_threshold) and (current_soc <= v2g_min_threshold):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 1
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        elif (soc_after_trip <= charge_threshold) and (current_soc < target_soc):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        elif (soc_after_trip <= charge_threshold) and (current_soc >= target_soc):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        else:
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        if aid not in participant_ids:
                            next_action = "Idle"
                            panelty = 0
                    elif (policy == "Gaussian") and (vpp_req == True):
                        if (soc_after_trip >= v2g_threshold) and (current_soc > v2g_min_threshold):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 1
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 3
                        elif (soc_after_trip >= v2g_threshold) and (current_soc <= v2g_min_threshold):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 1
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        elif (soc_after_trip <= charge_threshold) and (current_soc < target_soc):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        elif (soc_after_trip <= charge_threshold) and (current_soc >= target_soc):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        elif (soc_after_trip < v2g_threshold) and (soc_after_trip > charge_threshold):
                            if -1 <= action <= -0.5:
                                next_action = "V2G"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 2
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        else:
                            next_action = "Idle"
                            panelty = 3
                        if aid not in participant_ids:
                            next_action = "Idle"
                            panelty = 0
                    elif (policy == "Gaussian") and (vpp_req == False):
                        if (soc_after_trip >= v2g_threshold) and (current_soc > v2g_min_threshold):
                            if -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 1
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            elif 0.5 < action <= 1.0:
                                next_action = "Charging"
                                panelty = 2
                            else:
                                next_action = "Idle"
                                panelty = 2
                        elif (soc_after_trip >= v2g_threshold) and (current_soc <= v2g_min_threshold):
                            if -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 1
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            elif 0.5 < action <= 1.0:
                                next_action = "Charging"
                                panelty = 2
                            else:
                                next_action = "Idle"
                                panelty = 2
                        elif (soc_after_trip <= charge_threshold) and (current_soc < target_soc):
                            if -1.0 < action <= -0.5:
                                next_action = "Idle"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 3
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        elif (soc_after_trip <= charge_threshold) and (current_soc >= target_soc):
                            if 0.5 < action <= 1.0:
                                next_action = "Charging"
                                panelty = 2
                            else:
                                next_action = "Idle"
                                panelty = 2
                        elif (soc_after_trip < v2g_threshold) and (soc_after_trip > charge_threshold):
                            if -1.0 < action <= -0.5:
                                next_action = "Idle"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "Idle"
                                panelty = 3
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        else:
                            if -1.0 < action <= -0.5:
                                next_action = "Idle"
                                panelty = 3
                            elif -0.5 < action <= 0:
                                next_action = "V2H"
                                panelty = 3
                            elif 0 < action <= 0.5:
                                next_action = "Idle"
                                panelty = 2
                            else:
                                next_action = "Charging"
                                panelty = 2
                        if aid not in participant_ids:
                            next_action = "Idle"
                            panelty = 0
                    else:
                        if -1 <= action <= -0.5:
                            next_action = "V2G"
                            panelty = 3
                        elif -0.5 < action <= 0:
                            next_action = "V2H"
                            panelty = 2    
                        elif 0 < action <= 0.5:
                            next_action = "Idle"
                            panelty = 0
                        else:
                            next_action = "Charging"
                            panelty = 1
                        if aid not in participant_ids:
                            next_action = "Idle"
                            panelty = 0
                else:
                    next_action = "Idle"
                    panelty = 2

                # Table 2 (i) charge_only: V2H/V2G 만 금지, 충전은 허용
                if getattr(self, "disable_v2x", False) and next_action in ("V2G", "V2H"):
                    next_action = "Idle"

                V2H_power = home_power
                V2G_power = discharging_power - home_power
                V2G_amount = 0.0
                V2H_amount = 0.0
                CH_amount = 0.0
                N = 0.0
                                              
                if next_action == "Charging":
                    energy = charging_power #/ 4
                    temp_soc = current_soc + (energy / battery_kwh)
                    if temp_soc >= target_soc:
                        next_soc = target_soc
                        CH_amount = (target_soc - current_soc) * battery_kwh
                        V2H_amount = 0
                        V2G_amount = 0
                        reward_home = self.home_v2g_reward(next_action, V2G_amount, V2H_amount, CH_amount, prices[t_idx], self.incen_cfg, panelty, aid, step, self._home_load_hist, vpp_req, policy)
                    else:
                        next_soc = current_soc + (energy / battery_kwh)
                        CH_amount = energy
                        V2H_amount = 0
                        V2G_amount = 0
                        reward_home = self.home_v2g_reward(next_action, V2G_amount, V2H_amount, CH_amount, prices[t_idx], self.incen_cfg, panelty, aid, step, self._home_load_hist, vpp_req, policy)
                elif next_action == "Idle":
                    next_soc = current_soc
                    CH_amount = 0
                    V2H_amount = 0
                    V2G_amount = 0
                    reward_home = self.home_v2g_reward(next_action, V2G_amount, V2H_amount, CH_amount, prices[t_idx], self.incen_cfg, panelty, aid, step, self._home_load_hist, vpp_req, policy)
                elif next_action == "V2H":
                    energy = V2H_power #/ 4
                    temp_soc = current_soc - (energy / battery_kwh)
                    if temp_soc <= v2g_min_threshold:
                        next_soc = v2g_min_threshold
                        CH_amount = 0
                        V2H_amount = (v2g_min_threshold - temp_soc) * battery_kwh
                        V2G_amount = 0
                        reward_home = self.home_v2g_reward(next_action, V2G_amount, V2H_amount, CH_amount, prices[t_idx], self.incen_cfg, panelty, aid, step, self._home_load_hist, vpp_req, policy)
                    else:
                        next_soc = temp_soc
                        CH_amount = 0
                        V2H_amount = energy
                        V2G_amount = 0
                        reward_home = self.home_v2g_reward(next_action, V2G_amount, V2H_amount, CH_amount, prices[t_idx], self.incen_cfg, panelty, aid, step, self._home_load_hist, vpp_req, policy)
                elif next_action == "V2G":
                    energy = (V2H_power + V2G_power) #/ 4
                    temp_soc = current_soc - (energy / battery_kwh)
                    if temp_soc <= v2g_min_threshold:
                        next_soc = v2g_min_threshold
                        CH_amount = 0
                        V2H_amount = (v2g_min_threshold - temp_soc) * battery_kwh * V2H_power / (V2H_power + V2G_power)
                        V2G_amount = (v2g_min_threshold - temp_soc) * battery_kwh * V2G_power / (V2H_power + V2G_power)
                        reward_home = self.home_v2g_reward(next_action, V2G_amount, V2H_amount, CH_amount, prices[t_idx], self.incen_cfg, panelty, aid, step, self._home_load_hist, vpp_req, policy)
                    else:
                        next_soc = temp_soc
                        CH_amount = 0
                        V2H_amount = V2H_power #/ 4
                        V2G_amount = V2G_power #/ 4
                        reward_home = self.home_v2g_reward(next_action, V2G_amount, V2H_amount, CH_amount, prices[t_idx], self.incen_cfg, panelty, aid, step, self._home_load_hist, vpp_req, policy)
                
                rech_amount = 0
                
                if p_idx == 80:
                    if next_soc < soc_after_trip:
                        rech_amount = (target_soc - next_soc) * battery_kwh
                        re_CH_reward_home = rech_amount * prices[t_idx + 4]
                        next_soc = target_soc
                    else:
                        re_CH_reward_home = 0
                else:
                    re_CH_reward_home = 0                                       

                self._home_load_hist.setdefault(aid, {}).setdefault(step, []).append(float(home_power))
                
                N = 302.4 * (7 / battery_kwh) ** (-1.264)
                batt_deg_cost = (0.95 * 99 * battery_kwh) / (N * (7 / self.chg_cfg.cha_rate_kw))
                r_max_deg = batt_deg_cost if self.freeze_r_max else (
                    batt_deg_cost if self.use_batt_deg else 0.0)
                r_max = 1.2 * prices[t_idx] + 5.8 * float(self.VPP_incentive) - r_max_deg

                V2X_amount = float(V2H_amount + V2G_amount)
                if V2X_amount > 0:
                    N = 302.4 * (V2X_amount / battery_kwh) ** (-1.264)
                    batt_deg[aid] = (0.95 * 99 * battery_kwh) / (N * (V2X_amount + V2X_amount / self.chg_cfg.cha_rate_kw))
                else:
                    batt_deg[aid] = 0.0

                degradation_penalty = (
                    float(self.batt_deg_weight) * float(batt_deg[aid])
                    if self.use_batt_deg else 0.0)
                base_reward = float(reward_home) - float(re_CH_reward_home) - degradation_penalty
                
                temp_reward.append(base_reward)
                reward_by_agent[aid] = base_reward / r_max * self.re_w
                step_diagnostics[aid] = {
                    "t_idx": int(t_idx),
                    "day_idx": int(t_idx // self.step_day),
                    "step_of_day": int(step),
                    "hour": int(hour),
                    "agent_id": str(aid),
                    "algo": str(algo),
                    "policy": str(policy),
                    "stat": str(stat),
                    "use_batt_deg": bool(self.use_batt_deg),
                    "batt_deg_weight": float(self.batt_deg_weight),
                    "vpp_req": bool(vpp_req),
                    "price": float(prices[t_idx]),
                    "raw_action": float(action),
                    "executed_action": str(next_action),
                    "soc_before": float(current_soc),
                    "soc_after": float(next_soc),
                    "V2H_kWh": float(V2H_amount),
                    "V2G_kWh": float(V2G_amount),
                    "charge_kWh": float(CH_amount),
                    "recharge_kWh": float(rech_amount),
                    "economic_reward": float(reward_home),
                    "recharge_cost": float(re_CH_reward_home),
                    "degradation_cost": float(batt_deg[aid]),
                    "degradation_penalty_applied": float(degradation_penalty),
                    "reward_before_normalization": float(base_reward),
                    "reward_pre_grid": float(reward_by_agent[aid]),
                    "r_max": float(r_max),
                }
                
                
                V2H_by_agent[aid] = float(V2H_amount)
                V2G_by_agent[aid] = float(V2G_amount)
                CH_by_agent[aid] = float(CH_amount)
                RECH_by_agent[aid] = float(rech_amount)
                power_sum = float(V2G_amount + V2H_amount - CH_amount)
                
                storage_name = os.path.splitext(
                    os.path.basename(V2G_to_file[aid])
                )[0].lower()
                
                storage_multiplier[storage_name] = power_sum / 7
                
                real_action_by_agent[aid] = {
                    "t_idx": t_idx + 1,
                    "next_action": next_action, 
                    "current_soc": next_soc,                        
                }                       
        
        reward_values = [
            reward
            for aid, reward in reward_by_agent.items()
            if aid != "DERMS"
        ]
        n_agents = max(len(reward_values), 1)

        reward_mean = np.mean(reward_values) 
        next_t_idx = t_idx + 1
        (vpu, phase_vpu, voltage_metrics,) = self.dss_runner.solve_step_with_metrics(
            t_idx=t_idx,
            storage_multiplier=storage_multiplier,
            v_min=0.95,
            v_max=1.05,
        )
        self.last_vpu = vpu                      # Table 2 위반 개수 집계용
        self.last_voltage_metrics = voltage_metrics

        voltage_deviation = voltage_metrics.voltage_deviation
        violation_rate = voltage_metrics.violation_rate
        voltage_deviation_mean = voltage_deviation / 0.05
        vvio_mean = violation_rate / 1        
        grid_penalty = (
            lambda_vdev * voltage_deviation_mean
            + lambda_vio * vvio_mean
        )
        if self.grid_penalty_share == "divided":
            per_agent_penalty = grid_penalty / n_agents
        else:
            per_agent_penalty = grid_penalty
        per_agent_penalty *= float(self.grid_penalty_scale)

        for aid in reward_by_agent:
            if aid != "DERMS":
                reward_by_agent[aid] -= per_agent_penalty
                if aid in step_diagnostics:
                    step_diagnostics[aid]["grid_penalty"] = float(grid_penalty)
                    step_diagnostics[aid]["grid_penalty_applied"] = float(per_agent_penalty)
                    step_diagnostics[aid]["lambda_vdev"] = float(lambda_vdev)
                    step_diagnostics[aid]["lambda_vio"] = float(lambda_vio)
                    step_diagnostics[aid]["reward_final"] = float(reward_by_agent[aid])
                    step_diagnostics[aid]["voltage_deviation"] = float(voltage_deviation)
                    step_diagnostics[aid]["violation_rate"] = float(violation_rate)
                    self.ablation_history.append(step_diagnostics[aid])
        
        next_obs_by_agent = self.next_state(
            next_t_idx,
            real_action_by_agent,
            policy,
            ev_travel,
            vpp_req,
            ev_initial,
            prices,
            stat,
            voltage_deviation,
            violation_rate
        )
        
        sensitivity = {
            "base_reward_mean": reward_mean,
            "vdev_mean": voltage_deviation_mean,
            "vvio_mean": vvio_mean,
        }
        
        
        return next_obs_by_agent, reward_by_agent, V2H_by_agent, V2G_by_agent, CH_by_agent, RECH_by_agent, voltage_deviation, violation_rate, sensitivity
