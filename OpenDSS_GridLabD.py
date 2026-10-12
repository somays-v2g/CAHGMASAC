import os, re, glob
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import opendssdirect as dss
from pathlib import Path
from collections import defaultdict
from pathlib import Path



GRIDLABD_PATH = 'C:\\Program Files\\GridLAB-D 5.3.0\\bin\\gridlabd.exe'
BASE_DIR = Path.cwd()
BASE_PATH = BASE_DIR
LABD_OUTPUT_PATH = os.path.join(BASE_PATH, 'gridlabd_output')
os.makedirs(LABD_OUTPUT_PATH, exist_ok=True)
DATA_PATH = BASE_PATH / 'grid_input'
os.makedirs(DATA_PATH, exist_ok=True)
_master_candidates = [BASE_DIR / 'master_with_ev_pv_v2g.dss'] + sorted(BASE_DIR.glob('master_with_ev_pv_v2g*.dss'))
MASTER_DSS = next((p for p in _master_candidates if p.exists()), _master_candidates[0])
INPUT_DIR = BASE_DIR / 'csv_inputs'

START_DATE = '2026-01-01 00:00:00'
FREQ = '15min'

def parse_dss_files(base_path: str, parent_path: str) -> pd.DataFrame:
    dss_files = []
    for path in [base_path, parent_path]:
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith('.dss'):
                    dss_files.append(os.path.join(root, file))
    dss_files = list(set(dss_files))
    parsed_records = []
    for dss_file in dss_files:
        file_name = os.path.basename(dss_file)
        try:
            with open(dss_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()
        except Exception:
            continue
        clean_lines = []
        current_stmt = ''
        for line in raw_lines:
            line_str = line.strip()
            if not line_str or line_str.startswith('!') or line_str.startswith('//'):
                continue
            if line_str.startswith('~'):
                current_stmt += ' ' + line_str[1:].strip()
            else:
                if current_stmt:
                    clean_lines.append(current_stmt)
                current_stmt = line_str
        if current_stmt:
            clean_lines.append(current_stmt)
        found_in_file = 0
        for stmt in clean_lines:
            match_type = re.search('(?i)\\b(?:new|edit)\\s+(load|pvsystem|storage|generator)\\.([^\\s=]+)', stmt)
            if not match_type:
                continue
            asset_type = match_type.group(1).capitalize()
            asset_name = match_type.group(2)
            bus_match = re.search('(?i)\\bbus1\\s*=\\s*([^\\s,]+)', stmt)
            bus = bus_match.group(1).split('.')[0].strip('"\'') if bus_match else 'Unknown'
            kw_match = re.search('(?i)\\b(?:kW|Pmpp|kWRated|kVA|pf)\\s*=\\s*([\\d.]+)', stmt)
            rated_capacity = float(kw_match.group(1)) if kw_match else 1.0
            parsed_records.append({'asset_name': asset_name, 'type': asset_type, 'bus': bus, 'rated_capacity_kw': rated_capacity, 'file': file_name})
            found_in_file += 1
    return pd.DataFrame(parsed_records)

def match_asset_name(filename: str, dss_lookup: dict) -> dict:
    fn_lower = filename.lower()
    for key, info in dss_lookup.items():
        if key == fn_lower or key in fn_lower or fn_lower in key:
            return info
    fn_clean = re.sub('^(loadshape_|pvshape_|pv_|load_)', '', fn_lower)
    for key, info in dss_lookup.items():
        key_clean = re.sub('^(loadshape_|pvshape_|pv_|load_)', '', key)
        if fn_clean == key_clean or key_clean in fn_clean or fn_clean in key_clean:
            return info
    return None

def build_mapped_unified_dataframe(base_path: str, df_dss: pd.DataFrame) -> pd.DataFrame:
    patterns = ['EVAC_*_*.csv', 'EVDC_*.csv', 'loadshape_*.csv', 'pvshape_*.csv', 'V2G_*_*.csv']
    csv_files = []
    for pattern in patterns:
        csv_files.extend(glob.glob(os.path.join(base_path, pattern)))
    csv_files = sorted(set(csv_files))
    if not csv_files:
        return pd.DataFrame()
    sample_data = pd.read_csv(csv_files[0], header=None).values.flatten()
    data_length = len(sample_data)
    time_index = pd.date_range(start=START_DATE, periods=data_length, freq=FREQ)
    df_unified = pd.DataFrame(index=time_index)
    df_unified.index.name = 'timestamp'
    dss_lookup = {}
    if not df_dss.empty:
        for _, row in df_dss.iterrows():
            dss_lookup[row['asset_name'].lower()] = row.to_dict()
    success_count = 0
    mapped_count = 0
    for file_path in csv_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            data = pd.read_csv(file_path, header=None).values.flatten()
        except Exception as e:
            continue
        if len(data) != data_length:
            continue
        matched_info = match_asset_name(filename, dss_lookup)
        if matched_info:
            bus = matched_info['bus']
            asset_name = matched_info['asset_name']
            rated_kw = matched_info['rated_capacity_kw']
            col_name = f'Bus_{bus}_{asset_name}'
            df_unified[col_name] = data * rated_kw
            mapped_count += 1
        else:
            df_unified[filename] = data
        success_count += 1
    return df_unified

def export_gridlabd_players(df_subset: pd.DataFrame, output_dir: str, target_date: str):
    count = 0
    for col in df_subset.columns:
        player_filename = f'{target_date}_{col}.player'
        file_path = os.path.join(output_dir, player_filename)
        with open(file_path, 'w') as f:
            for ts, val in df_subset[col].items():
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S PST')
                f.write(f'{ts_str},{val:.4f}\n')
        count += 1

MAX_STEPS = 480
INTERVAL_H = 0.25
TARGET_BUS = '63.1'
SELECT_BUSES = ['63.1']

def _token(fname, prefix_regex):
    base = os.path.splitext(os.path.basename(fname))[0]
    return re.sub(prefix_regex, '', base, flags=re.I).lower()
pat_load = os.path.join(INPUT_DIR, '[lL][oO][aA][dD][sS][hH][aA][pP][eE]_*.c[sS][vV]')
pat_evac = os.path.join(INPUT_DIR, '[eE][vV][aA][cC]_*.c[sS][vV]')
pat_pv = os.path.join(INPUT_DIR, '[pP][vV][sS][hH][aA][pP][eE]_*.c[sS][vV]')
pat_v2g = os.path.join(INPUT_DIR, '[vV][2][gG]_*.c[sS][vV]')
load_map = {_token(p, '^loadshape_'): p for p in glob.glob(pat_load)}
evac_map = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in glob.glob(pat_evac)}
pv_map = {_token(p, '^pvshape_'): p for p in glob.glob(pat_pv)}
v2g_map = {_token(p, ''): p for p in glob.glob(pat_v2g)}
if len(load_map) == 0 and len(evac_map) == 0 and (len(pv_map) == 0):
    raise SystemExit(f'CSV를 찾지 못했습니다. DATA_DIR 확인: {INPUT_DIR}')

def read_ratio_series(path):
    return pd.read_csv(path, header=None).iloc[:, 0].astype(float).values
load_profiles = {k: read_ratio_series(v) for k, v in load_map.items()}
evac_profiles = {k: read_ratio_series(v) for k, v in evac_map.items()}
pv_profiles = {k: read_ratio_series(v) for k, v in pv_map.items()}
v2g_profiles = {k: read_ratio_series(v) for k, v in v2g_map.items()}
csv_len_candidates = []
if load_profiles:
    csv_len_candidates.append(len(next(iter(load_profiles.values()))))
if evac_profiles:
    csv_len_candidates.append(len(next(iter(evac_profiles.values()))))
if pv_profiles:
    csv_len_candidates.append(len(next(iter(pv_profiles.values()))))
if v2g_profiles:
    csv_len_candidates.append(len(next(iter(v2g_profiles.values()))))
if not csv_len_candidates:
    raise SystemExit('프로파일이 비어 있습니다. CSV 파일을 확인하세요.')
csv_len = min(csv_len_candidates)
NPTS = min(csv_len, MAX_STEPS if MAX_STEPS else csv_len)

def setup_circuit():
    """OpenDSS 초기화 및 기본값 스냅샷"""
    dss.Basic.ClearAll()
    dss.Text.Command(f'compile [{MASTER_DSS}]')
    dss.Text.Command('set controlmode=static')
    dss.Text.Command('set mode=snap')
    if dss.Error.Number() != 0:
        raise RuntimeError(dss.Error.Description())
    dss.Text.Command('set loadmult=1.0')
    for ld in dss.Loads.AllNames():
        dss.Loads.Name(ld)
        dss.Loads.Daily('')
        dss.Loads.Yearly('')
        dss.Loads.Duty('')
        dss.Loads.Growth('')
    load_names = [ld.lower() for ld in dss.Loads.AllNames()]
    pv_names = [pv.lower() for pv in dss.PVsystems.AllNames()]
    storage_names = [st.lower() for st in dss.Storages.AllNames()]
    base_load_kw, base_load_kvar = ({}, {})
    for ld in dss.Loads.AllNames():
        dss.Loads.Name(ld)
        base_load_kw[ld.lower()] = dss.Loads.kW()
        base_load_kvar[ld.lower()] = dss.Loads.kvar()
    base_pv_pmpp = {}
    for pv in dss.PVsystems.AllNames():
        dss.PVsystems.Name(pv)
        base_pv_pmpp[pv.lower()] = dss.PVsystems.Pmpp()
    evac_elems = sorted(set(evac_profiles.keys()) & set(load_names))
    if len(evac_elems) == 0 and evac_profiles:
        pass
    return (load_names, pv_names, storage_names, base_load_kw, base_load_kvar, base_pv_pmpp, evac_elems)

def get_bus_and_node(s: str):
    s = str(s).strip()
    if '.' in s:
        b, n = s.split('.', 1)
        try:
            return (b, int(n))
        except:
            return (b, None)
    return (s, None)

def read_bus_node_vpu(bus_dot_node: str):
    bname, node = get_bus_and_node(bus_dot_node)
    dss.Circuit.SetActiveBus(bname)
    arr = dss.Bus.puVmagAngle()
    if not arr:
        return np.nan
    if node is None:
        return float(np.max(arr[0::2]))
    nodes = list(dss.Bus.Nodes())
    if node in nodes:
        k = nodes.index(node)
        return float(arr[2 * k])
    return float(np.max(arr[0::2]))

def _set_storage(dss, st, p_kw, q_kvar):
    dss.Storages.Name(st)
    if p_kw > 0:
        dss.Text.Command(f'edit storage.{st} state=Discharging kW={p_kw * 7} kvar={q_kvar}')
    elif p_kw < 0:
        dss.Text.Command(f'edit storage.{st} state=Charging kW={abs(p_kw) * 7} kvar={q_kvar}')
    else:
        dss.Text.Command(f'edit storage.{st} state=Idling kW=0 kvar=0')
    return

def parse_bus_and_nodes_list(s: str):
    s = str(s).strip()
    if '.' not in s:
        return (s, None)
    parts = s.split('.')
    b = parts[0]
    nodes = []
    for p in parts[1:]:
        try:
            nodes.append(int(p))
        except:
            pass
    if len(nodes) == 0:
        nodes = None
    return (b, nodes)

def read_bus_node_vpu(bus_dot_node: str):
    b, nodes = parse_bus_and_nodes_list(bus_dot_node)
    vals = read_bus_phase_vpus(b)
    if not vals:
        return np.nan
    if nodes is None:
        return float(np.max([v for _, v in vals]))
    return float(np.max([v for n, v in vals if n in set(nodes)]) if any((n in set(nodes) for n, _ in vals)) else np.max([v for _, v in vals]))

def read_bus_phase_vpus(bus_name: str):
    dss.Circuit.SetActiveBus(bus_name)
    arr = dss.Bus.puVmagAngle()
    if not arr:
        return []
    nodes = list(dss.Bus.Nodes())
    vpus = arr[0::2]
    return list(zip(nodes, map(float, vpus)))

def apply_scenario_at_step(t, scenario_id, load_names, pv_names, storage_names, base_load_kw, base_load_kvar, base_pv_pmpp, evac_elems, v2g_profiles=None, v2g_kvar_profiles=None):
    for ld in dss.Loads.AllNames():
        name = ld.lower()
        dss.Loads.Name(ld)
        mult_series = load_profiles.get(name, None)
        m = float(mult_series[t]) if mult_series is not None else 1.0
        dss.Loads.kW(base_load_kw[name] * m)
        dss.Loads.kvar(base_load_kvar[name] * m)
    if scenario_id >= 2 and evac_profiles:
        for ev in evac_elems:
            dss.Loads.Name(ev)
            emult = float(evac_profiles[ev][t])
            dss.Loads.kW(base_load_kw[ev] * emult)
            dss.Loads.kvar(base_load_kvar[ev] * emult)
    else:
        for ev in evac_elems:
            dss.Loads.Name(ev)
            dss.Loads.kW(0.0)
            dss.Loads.kvar(0.0)
    if scenario_id in (3, 4) and pv_profiles:
        for pv in pv_profiles.keys():
            if pv in pv_names:
                dss.PVsystems.Name(pv)
                pmult = float(pv_profiles[pv][t])
                dss.PVsystems.Pmpp(base_pv_pmpp[pv] * pmult)
    else:
        for pv in pv_profiles.keys():
            if pv in pv_names:
                dss.PVsystems.Name(pv)
                dss.PVsystems.Pmpp(0.0)
    if scenario_id == 4 and storage_names:
        for st in storage_names:
            p_series = v2g_profiles.get(st)
            q_series = v2g_kvar_profiles.get(st) if v2g_kvar_profiles else None
            p_kw = float(p_series[t]) if p_series is not None else 0.0
            q_kvar = float(q_series[t]) if q_series is not None else 0.0
            _set_storage(dss, st, p_kw, q_kvar)
    else:
        for st in storage_names:
            dss.Storages.Name(st)
            dss.CktElement.Enabled(0)

def set_storages_enabled(enabled: bool, storage_names):
    for st in storage_names:
        dss.Storages.Name(st)
        dss.CktElement.Enabled(1 if enabled else 0)

def neutralize_storages(storage_names):
    for st in storage_names:
        dss.Text.Command(f'edit storage.{st} state=Idling kW=0 kvar=0 %Idlingkvar=0')

def run_scenario_collect(scenario_id, selected_bus_list):
    load_names, pv_names, storage_names, base_load_kw, base_load_kvar, base_pv_pmpp, evac_elems = setup_circuit()
    stor_on = scenario_id == 4
    set_storages_enabled(stor_on, storage_names)
    if stor_on:
        neutralize_storages(storage_names)
    expanded_phase_keys = []
    normalized_sel = []
    for raw in [s.strip() for s in selected_bus_list]:
        b, nodes = parse_bus_and_nodes_list(raw)
        dss.Circuit.SetActiveBus(b)
        bus_nodes = list(dss.Bus.Nodes())
        if not bus_nodes:
            continue
        if nodes is None:
            use_nodes = bus_nodes
        else:
            use_nodes = [n for n in nodes if n in set(bus_nodes)]
        for n in use_nodes:
            expanded_phase_keys.append(f'{b}.{n}')
        normalized_sel.append(raw)
    sel_dict = {key: np.zeros(NPTS) for key in normalized_sel}
    phase_sel_dict = {key: np.zeros(NPTS) for key in expanded_phase_keys}
    target_series = np.zeros(NPTS)
    for t in range(NPTS):
        apply_scenario_at_step(t, scenario_id, load_names, pv_names, storage_names, base_load_kw, base_load_kvar, base_pv_pmpp, evac_elems, v2g_profiles, v2g_kvar_profiles=None)
        dss.Solution.Solve()
        if dss.Error.Number() != 0:
            raise RuntimeError(dss.Error.Description())
        target_series[t] = read_bus_node_vpu(TARGET_BUS)
        for ph_key in expanded_phase_keys:
            b, nodes = parse_bus_and_nodes_list(ph_key)
            node = nodes[0]
            vals = read_bus_phase_vpus(b)
            v = next((v for n, v in vals if n == node), np.nan)
            phase_sel_dict[ph_key][t] = v
        for raw in normalized_sel:
            b, nodes = parse_bus_and_nodes_list(raw)
            vals = read_bus_phase_vpus(b)
            if not vals:
                sel_dict[raw][t] = np.nan
                continue
            if nodes is None:
                sel_dict[raw][t] = float(np.max([v for _, v in vals]))
            else:
                cand = [v for n, v in vals if n in set(nodes)]
                sel_dict[raw][t] = float(np.max(cand)) if cand else float(np.max([v for _, v in vals]))
        if (t + 1) % 100 == 0:
            pass
    return (target_series, sel_dict, phase_sel_dict)

def find_lowest_k_phase_series(phase_dict, k=10):
    rows = []
    for key, series in phase_dict.items():
        if series is None or len(series) == 0:
            continue
        v = np.array(series, dtype=float)
        if np.all(np.isnan(v)):
            continue
        idx = int(np.nanargmin(v))
        rows.append({'key': key, 'min_vpu': float(v[idx]), 't_idx': idx, 't_hour': float(idx * INTERVAL_H)})
    rows.sort(key=lambda r: r['min_vpu'])
    return rows[:k]

def plot_lowest_k_phase_series(phase_dict, time_axis, k=10, title_prefix='Lowest phase voltages'):
    topk = find_lowest_k_phase_series(phase_dict, k=k)
    keys = [r['key'] for r in topk]
    plt.figure(figsize=(12, 6))
    for key in keys:
        plt.plot(time_axis, phase_dict[key], label=key)
    plt.axhline(1.05, linestyle='--', linewidth=1, alpha=0.7, label='Upper 1.05')
    plt.axhline(0.95, linestyle='--', linewidth=1, alpha=0.7, label='Lower 0.95')
    plt.xlabel('시간 [h]')
    plt.ylabel('Vpu')
    plt.title(f'{title_prefix} — top {k} phases (min over time)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
    for i, r in enumerate(topk, 1):
        pass
time = np.arange(NPTS) * INTERVAL_H
'\nprint("[INFO] 시나리오 1 (Load only) 실행")\nvpu_s1, sel_s1 = run_scenario_collect(1, SELECT_BUSES)\n\nprint("[INFO] 시나리오 2 (Load + EVAC) 실행")\nvpu_s2, sel_s2 = run_scenario_collect(2, SELECT_BUSES)\n\nprint("[INFO] 시나리오 3 (Load + EVAC + PV) 실행")\nvpu_s3, sel_s3 = run_scenario_collect(3, SELECT_BUSES)\n\nprint("[INFO] 시나리오 4 (Load + EVAC + PV + V2G) 실행")\nvpu_s4, sel_s4 = run_scenario_collect(4, SELECT_BUSES)\n'
vpu_s3, sel_s3, phase_s3 = run_scenario_collect(3, SELECT_BUSES)
vpu_s4, sel_s4, phase_s4 = run_scenario_collect(4, SELECT_BUSES)


BASE_DIR = Path.cwd()
INPUT_DIR = BASE_DIR / 'csv_inputs'
START_DATE = '2026-01-01 00:00:00'
FREQ = '15min'
MAX_STEPS = 480
GLD_SCENARIO = 4
TARGET_BUS = '63'
TARGET_PHASE = 'A'
PF_DEFAULT = 0.95
PLAYER_DIR = BASE_DIR / 'gridlabd_busphase_players'
GLM_FILE = BASE_DIR / 'ieee123_from_dss.glm'
OUTPUT_CSV = BASE_DIR / 'output_node63_volts.csv'

def first_existing(patterns: list[str]) -> Path:
    for pat in patterns:
        hits = sorted(BASE_DIR.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f'Cannot find any of: {patterns} in {BASE_DIR}')
MASTER_DSS = first_existing(['master_with_ev_pv_v2g.dss', 'master_with_ev_pv_v2g*.dss'])
LINECODE_DSS = first_existing(['IEEELinecodes.dss', 'IEEELinecodes*.dss'])
LOAD_DSS = first_existing(['IEEE123Loads_pf95.dss', 'IEEE123Loads_pf95*.dss'])
EV_DSS = first_existing(['ev_additions_pf95.dss', 'ev_additions_pf95*.dss'])
PV_DSS = first_existing(['IEEE123Pv.dss', 'IEEE123Pv*.dss'])
V2G_DSS = first_existing(['IEEE123V2Gs.dss', 'IEEE123V2Gs*.dss'])
REG_DSS = first_existing(['IEEE123Regulators.dss', 'IEEE123Regulators*.dss'])

def strip_comment(line: str) -> str:
    return line.split('!', 1)[0].strip()

def join_dss_commands(path: Path) -> list[str]:
    commands, current = ([], '')
    for raw in path.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        line = strip_comment(raw)
        if not line:
            continue
        if line.startswith('~'):
            current += ' ' + line[1:].strip()
        else:
            if current:
                commands.append(current)
            current = line
    if current:
        commands.append(current)
    return commands
OBJ_RE = re.compile('^\\s*(?:new|edit)\\s+(?P<cls>[A-Za-z0-9_]+)\\.(?P<name>[^\\s,]+)', re.I)
PROP_RE = re.compile('(?P<key>%?[A-Za-z][A-Za-z0-9_%]*)\\s*=\\s*(?P<val>\\[[^\\]]*\\]|\\([^\\)]*\\)|\\"[^\\"]*\\"|\'[^\']*\'|[^\\s]+)', re.I)

def props_of(command: str) -> dict[str, str]:
    return {m.group('key').lower(): m.group('val').strip().strip(',;') for m in PROP_RE.finditer(command)}

def parse_objects(path: Path, cls: str) -> list[dict]:
    out = []
    for cmd in join_dss_commands(path):
        m = OBJ_RE.match(cmd)
        if not m or m.group('cls').lower() != cls.lower():
            continue
        out.append({'name': m.group('name').strip('"\''), 'props': props_of(cmd), 'command': cmd})
    return out

def fnum(v, default=0.0):
    if v is None:
        return default
    try:
        return float(str(v).strip().strip('"\''))
    except Exception:
        return default

def parse_bus_spec(spec: str, phases_hint: int | None=None) -> tuple[str, list[int]]:
    s = str(spec).strip().strip('"\'').strip('[]')
    parts = s.split('.')
    bus = parts[0]
    nodes = []
    for x in parts[1:]:
        try:
            n = int(x)
            if n in (1, 2, 3):
                nodes.append(n)
        except Exception:
            pass
    if not nodes:
        if phases_hint == 1:
            nodes = [1]
        elif phases_hint == 2:
            nodes = [1, 2]
        else:
            nodes = [1, 2, 3]
    return (bus, nodes)
PHASE_NAME = {1: 'A', 2: 'B', 3: 'C'}

def channel_for_device(props: dict) -> tuple[str, list[str], str]:
    phases = int(fnum(props.get('phases'), 3))
    conn = str(props.get('conn', 'wye')).lower()
    bus, nodes = parse_bus_spec(props.get('bus1', props.get('bus', '')), phases)
    if phases >= 3:
        return (bus, ['A', 'B', 'C'], conn)
    if conn.startswith('delta') and len(nodes) >= 2:
        pair = tuple(nodes[:2])
        pair_map = {(1, 2): 'AB', (2, 3): 'BC', (3, 1): 'CA', (2, 1): 'AB', (3, 2): 'BC', (1, 3): 'CA'}
        return (bus, [pair_map[pair]], conn)
    return (bus, [PHASE_NAME[nodes[0]]], conn)

def read_profile(path: Path, npts: int | None=None) -> np.ndarray:
    a = pd.read_csv(path, header=None).iloc[:, 0]
    a = pd.to_numeric(a, errors='coerce').dropna().to_numpy(float)
    return a[:npts] if npts else a

def find_profile(stem: str) -> Path | None:
    key = stem.lower() + '.csv'
    for p in INPUT_DIR.glob('*.csv'):
        if p.name.lower() == key:
            return p
    return None

def determine_npts() -> int:
    lengths = []
    for p in INPUT_DIR.glob('*.csv'):
        name = p.name.lower()
        if name.startswith(('loadshape_', 'evac_', 'pvshape_', 'v2g_')):
            try:
                lengths.append(len(read_profile(p)))
            except Exception:
                pass
    if not lengths:
        raise FileNotFoundError(f'No load/EV/PV/V2G CSV profiles found in {INPUT_DIR}')
    n = min(lengths)
    if MAX_STEPS:
        n = min(n, int(MAX_STEPS))
    return n

def topology_bus_phases() -> dict[str, set[int]]:
    bus_phases: dict[str, set[int]] = defaultdict(set)
    for ln in collect_lines():
        bus_phases[ln['b1']].update(ln['phases'])
        bus_phases[ln['b2']].update(ln['phases'])
    for r in collect_regulators():
        bus_phases[r['from']].update(r['nodes'])
        bus_phases[r['to']].update(r['nodes'])
    return bus_phases

def channel_nodes(channel: str) -> set[int]:
    m = {'A': 1, 'B': 2, 'C': 3}
    return {m[c] for c in channel if c in m}

def device_phases_valid(bus: str, channels: list[str], topo: dict[str, set[int]]) -> tuple[bool, set[int], set[int]]:
    requested = set()
    for ch in channels:
        requested.update(channel_nodes(ch))
    available = set(topo.get(bus, set()))
    return (bool(available) and requested.issubset(available), requested, available)

def build_busphase_players() -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame, int]:
    npts = determine_npts()
    net: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(npts, dtype=complex))
    rows = []
    topo = topology_bus_phases()
    mismatch_rows = []

    def reject_if_phase_mismatch(bus: str, channels: list[str], source: str, kind: str) -> bool:
        ok, requested, available = device_phases_valid(bus, channels, topo)
        if ok:
            return False
        mismatch_rows.append({'kind': kind, 'source': source, 'bus': bus, 'requested_phases': ''.join((PHASE_NAME[n] for n in sorted(requested))), 'available_phases': ''.join((PHASE_NAME[n] for n in sorted(available))), 'action': 'SKIPPED_ENTIRE_DEVICE'})
        return True

    def add(bus: str, channel: str, p_kw: np.ndarray, q_kvar: np.ndarray, source: str, kind: str):
        net[bus, channel] += p_kw + 1j * q_kvar
        rows.append({'bus': bus, 'channel': channel, 'kind': kind, 'source': source, 'p_kw_min': float(np.nanmin(p_kw)), 'p_kw_max': float(np.nanmax(p_kw)), 'q_kvar_min': float(np.nanmin(q_kvar)), 'q_kvar_max': float(np.nanmax(q_kvar))})
    for obj in parse_objects(LOAD_DSS, 'load'):
        name, pr = (obj['name'], obj['props'])
        bus, channels, _ = channel_for_device(pr)
        if reject_if_phase_mismatch(bus, channels, name, 'base_load'):
            continue
        p0, q0 = (fnum(pr.get('kw')), fnum(pr.get('kvar')))
        pf = find_profile(f'loadshape_{name}')
        mult = read_profile(pf, npts) if pf else np.ones(npts)
        split = len(channels)
        for ch in channels:
            add(bus, ch, p0 * mult / split, q0 * mult / split, name, 'base_load')
    if GLD_SCENARIO >= 2:
        for obj in parse_objects(EV_DSS, 'load'):
            name, pr = (obj['name'], obj['props'])
            pf = find_profile(name)
            if pf is None:
                continue
            bus, channels, _ = channel_for_device(pr)
            if reject_if_phase_mismatch(bus, channels, name, 'EV'):
                continue
            p0, q0 = (fnum(pr.get('kw')), fnum(pr.get('kvar')))
            mult = read_profile(pf, npts)
            split = len(channels)
            for ch in channels:
                add(bus, ch, p0 * mult / split, q0 * mult / split, name, 'EV')
    if GLD_SCENARIO in (3, 4):
        for obj in parse_objects(PV_DSS, 'pvsystem'):
            name, pr = (obj['name'], obj['props'])
            profile_key = str(pr.get('yearly', f'pvshape_{name}')).strip('"\'')
            pf = find_profile(profile_key)
            if pf is None:
                continue
            bus, channels, _ = channel_for_device(pr)
            if reject_if_phase_mismatch(bus, channels, name, 'PV'):
                continue
            pmpp = fnum(pr.get('pmpp'), fnum(pr.get('kw'), fnum(pr.get('kva'))))
            mult = read_profile(pf, npts)
            split = len(channels)
            for ch in channels:
                p = -(pmpp * mult / split)
                add(bus, ch, p, np.zeros(npts), name, 'PV')
    if GLD_SCENARIO == 4:
        for obj in parse_objects(V2G_DSS, 'storage'):
            name, pr = (obj['name'], obj['props'])
            pf = find_profile(name)
            if pf is None:
                continue
            bus, channels, _ = channel_for_device(pr)
            if reject_if_phase_mismatch(bus, channels, name, 'V2G'):
                continue
            rated_kw = fnum(pr.get('kwrated'), 7.0)
            cmd = read_profile(pf, npts)
            split = len(channels)
            for ch in channels:
                p = -(rated_kw * cmd / split)
                add(bus, ch, p, np.zeros(npts), name, 'V2G')
    for obj in parse_objects(MASTER_DSS, 'capacitor'):
        name, pr = (obj['name'], obj['props'])
        bus, channels, _ = channel_for_device(pr)
        if reject_if_phase_mismatch(bus, channels, name, 'capacitor'):
            continue
        kvar = fnum(pr.get('kvar'))
        split = len(channels)
        for ch in channels:
            add(bus, ch, np.zeros(npts), np.full(npts, -kvar / split), name, 'capacitor')
    mismatch_df = pd.DataFrame(mismatch_rows)
    mismatch_path = BASE_DIR / 'gridlabd_phase_mismatch_report.csv'
    mismatch_df.to_csv(mismatch_path, index=False)
    PLAYER_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(START_DATE, periods=npts, freq=FREQ)
    for (bus, ch), s in sorted(net.items()):
        out = PLAYER_DIR / f'bus_{bus}_{ch}.csv'
        with out.open('w', encoding='utf-8') as f:
            for ts, z in zip(timestamps, s):
                z_w = z * 1000.0
                sign = '+' if z_w.imag >= 0 else '-'
                f.write(f'{ts:%Y-%m-%d %H:%M:%S},{z_w.real:.6f}{sign}{abs(z_w.imag):.6f}j\n')
    summary = pd.DataFrame(rows)
    summary.to_csv(BASE_DIR / 'gridlabd_busphase_mapping.csv', index=False)
    return (net, summary, npts)

def parse_lower_triangle(text: str) -> np.ndarray:
    s = str(text).strip().strip('[]()')
    rows = [r.strip() for r in s.split('|') if r.strip()]
    vals = [[float(x) for x in re.split('[ ,]+', r.strip()) if x] for r in rows]
    n = len(vals)
    M = np.zeros((n, n), float)
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            M[i, j] = v
            M[j, i] = v
    return M

def read_linecodes() -> dict[str, dict]:
    out = {}
    for obj in parse_objects(LINECODE_DSS, 'linecode'):
        pr = obj['props']
        nph = int(fnum(pr.get('nphases'), 3))
        R = parse_lower_triangle(pr.get('rmatrix', '[0]'))
        X = parse_lower_triangle(pr.get('xmatrix', '[0]'))
        C = parse_lower_triangle(pr.get('cmatrix', '[0]')) if 'cmatrix' in pr else np.zeros_like(R)
        out[obj['name'].lower()] = {'nph': nph, 'R': R, 'X': X, 'C': C}
    return out

def gld_name(bus: str) -> str:
    return 'node_' + re.sub('[^A-Za-z0-9_]', '_', bus)

def phase_string(nodes: list[int], include_neutral=True) -> str:
    s = ''.join((PHASE_NAME[n] for n in (1, 2, 3) if n in nodes))
    return s + ('N' if include_neutral else '')

def collect_lines() -> list[dict]:
    lines = []
    for obj in parse_objects(MASTER_DSS, 'line'):
        pr = obj['props']
        phases = int(fnum(pr.get('phases'), 3))
        b1, p1 = parse_bus_spec(pr.get('bus1', ''), phases)
        b2, p2 = parse_bus_spec(pr.get('bus2', ''), phases)
        lines.append({'name': obj['name'], 'props': pr, 'b1': b1, 'b2': b2, 'phases': p1})
    return lines

def resolve_like(objects: list[dict]) -> dict[str, dict]:
    raw = {o['name'].lower(): dict(o['props']) for o in objects}
    resolved = {}

    def one(k, stack=None):
        if k in resolved:
            return resolved[k]
        stack = stack or set()
        if k in stack:
            return raw[k]
        stack.add(k)
        p = dict(raw[k])
        parent = p.get('like')
        if parent and parent.lower() in raw:
            base = dict(one(parent.lower(), stack))
            base.update({kk: vv for kk, vv in p.items() if kk != 'like'})
            p = base
        resolved[k] = p
        return p
    for k in raw:
        one(k)
    return resolved

def collect_regulators() -> list[dict]:
    tf_objs = parse_objects(MASTER_DSS, 'transformer') + parse_objects(REG_DSS, 'transformer')
    rc_objs = parse_objects(MASTER_DSS, 'regcontrol') + parse_objects(REG_DSS, 'regcontrol')
    tfs = resolve_like(tf_objs)
    rcs = resolve_like(rc_objs)
    regs = []
    for rc_name, rc in rcs.items():
        tf_name = str(rc.get('transformer', '')).lower()
        if tf_name not in tfs:
            continue
        tf = tfs[tf_name]
        buses = str(tf.get('buses', '')).strip('[]').split()
        if len(buses) < 2:
            continue
        ph = int(fnum(tf.get('phases'), 3))
        b1, nodes = parse_bus_spec(buses[0], ph)
        b2, _ = parse_bus_spec(buses[1], ph)
        regs.append({'name': tf_name, 'from': b1, 'to': b2, 'nodes': nodes, 'rc': rc})
    return regs

def build_glm(net: dict[tuple[str, str], np.ndarray], npts: int) -> str:
    linecodes = read_linecodes()
    lines = collect_lines()
    regs = collect_regulators()
    bus_phases: dict[str, set[int]] = defaultdict(set)
    for ln in lines:
        bus_phases[ln['b1']].update(ln['phases'])
        bus_phases[ln['b2']].update(ln['phases'])
    for r in regs:
        bus_phases[r['from']].update(r['nodes'])
        bus_phases[r['to']].update(r['nodes'])
    stop = pd.Timestamp(START_DATE) + pd.to_timedelta((npts + 1) * 15, unit='m')
    parts = [f"clock {{\n    timezone EST5EDT;\n    starttime '{pd.Timestamp(START_DATE):%Y-%m-%d %H:%M:%S}';\n    stoptime '{stop:%Y-%m-%d %H:%M:%S}';\n}}\n\nmodule powerflow {{\n    solver_method NR;\n    default_maximum_voltage_error 1e-4;\n}};\nmodule tape;\n"]
    for bus in sorted(bus_phases, key=lambda x: (not x.replace('.', '').isdigit(), x)):
        if bus.upper().endswith('_OPEN'):
            continue
        nodes = sorted(bus_phases[bus]) or [1, 2, 3]
        phases = phase_string(nodes)
        nominal = 480.0 / np.sqrt(3) if bus == '610' else 4160.0 / np.sqrt(3)
        cls = 'substation' if bus == '150' else 'node'
        swing = '\n    bustype SWING;' if bus == '150' else ''
        parts.append(f'object {cls} {{\n    name {gld_name(bus)};\n    phases {phases};\n    nominal_voltage {nominal:.6f};{swing}\n}}\n')
    for r in regs:
        rc = r['rc']
        nodes = r['nodes']
        phases = phase_string(nodes)
        bank = len(nodes) == 3
        ct_phase = PHASE_NAME[nodes[0]]
        cfg = f"regcfg_{re.sub('[^A-Za-z0-9_]', '_', r['name'])}"
        obj = f"reg_{re.sub('[^A-Za-z0-9_]', '_', r['name'])}"
        R = fnum(rc.get('r'), 0)
        X = fnum(rc.get('x'), 0)
        compR = '\n'.join((f'    compensator_r_setting_{PHASE_NAME[n]} {R};' for n in nodes))
        compX = '\n'.join((f'    compensator_x_setting_{PHASE_NAME[n]} {X};' for n in nodes))
        parts.append(f'''object regulator_configuration {{\n    name {cfg};\n    connect_type WYE_WYE;\n    band_center {fnum(rc.get('vreg'), 120)};\n    band_width {fnum(rc.get('band'), 2)};\n    time_delay 30;\n    dwell_time 15;\n    raise_taps 16;\n    lower_taps 16;\n    current_transducer_ratio {fnum(rc.get('ctprim'), 50)};\n    power_transducer_ratio {fnum(rc.get('ptratio'), 20)};\n{compR}\n{compX}\n    CT_phase "{ct_phase}";\n    PT_phase "{ct_phase}";\n    regulation 0.10;\n    Control LINE_DROP_COMP;\n    control_level {('BANK' if bank else 'INDIVIDUAL')};\n    Type A;\n    tap_pos_A 0;\n    tap_pos_B 0;\n    tap_pos_C 0;\n}}\nobject regulator {{\n    name {obj};\n    phases {phases};\n    from {gld_name(r['from'])};\n    to {gld_name(r['to'])};\n    configuration {cfg};\n}}\n''')
    for ln in lines:
        name, pr = (ln['name'], ln['props'])
        if name.lower() in {'sw7', 'sw8'} or ln['b2'].upper().endswith('_OPEN'):
            continue
        phases = phase_string(ln['phases'])
        cfg = f"cfg_{re.sub('[^A-Za-z0-9_]', '_', name)}"
        linecode = str(pr.get('linecode', '')).lower()
        Z = np.zeros((3, 3), complex)
        C = np.zeros((3, 3), float)
        if linecode in linecodes:
            lc = linecodes[linecode]
            src_nodes = ln['phases']
            for i, pi in enumerate(src_nodes):
                for j, pj in enumerate(src_nodes):
                    if i < lc['R'].shape[0] and j < lc['R'].shape[1]:
                        Z[pi - 1, pj - 1] = (lc['R'][i, j] + 1j * lc['X'][i, j]) * 5.28
                        if i < lc['C'].shape[0] and j < lc['C'].shape[1]:
                            C[pi - 1, pj - 1] = lc['C'][i, j] * 5.28
        else:
            r1 = fnum(pr.get('r1'), 0.001)
            x1 = fnum(pr.get('x1'), 0.0)
            for pi in ln['phases']:
                Z[pi - 1, pi - 1] = (r1 + 1j * x1) * 5.28
        zprops = []
        cprops = []
        for i in range(3):
            for j in range(3):
                if Z[i, j] != 0:
                    sign = '+' if Z[i, j].imag >= 0 else '-'
                    zprops.append(f'    z{i + 1}{j + 1} {Z[i, j].real:.9f}{sign}{abs(Z[i, j].imag):.9f}j;')
                if C[i, j] != 0:
                    cprops.append(f'    c{i + 1}{j + 1} {C[i, j]:.9f};')
        parts.append('object line_configuration {\n    name ' + cfg + ';\n' + '\n'.join(zprops + cprops) + '\n}\n')
        length_ft = fnum(pr.get('length'), 0.001) * 1000.0
        parts.append(f"object overhead_line {{\n    name line_{re.sub('[^A-Za-z0-9_]', '_', name)};\n    phases {phases};\n    from {gld_name(ln['b1'])};\n    to {gld_name(ln['b2'])};\n    length {length_ft:.6f} ft;\n    configuration {cfg};\n}}\n")
    grouped: dict[str, list[str]] = defaultdict(list)
    for bus, ch in sorted(net):
        req = channel_nodes(ch)
        avail = set(bus_phases.get(bus, set()))
        if not req.issubset(avail):
            continue
        grouped[bus].append(ch)
    for bus, channels in grouped.items():
        props = []
        for ch in channels:
            player = (PLAYER_DIR / f'bus_{bus}_{ch}.csv').as_posix()
            props.append(f'    object player {{\n        file "{player}";\n        property constant_power_{ch};\n    }};')
        active_letters = sorted(set((c for ch in channels for c in ch if c in 'ABC')), key='ABC'.index)
        ph = ''.join(active_letters) + 'N'
        nominal = 480.0 / np.sqrt(3) if bus == '610' else 4160.0 / np.sqrt(3)
        parts.append(f"object load {{\n    name load_{re.sub('[^A-Za-z0-9_]', '_', bus)};\n    parent {gld_name(bus)};\n    phases {ph};\n    nominal_voltage {nominal:.6f};\n{os.linesep.join(props)}\n}}\n")
    target_node = gld_name(TARGET_BUS)
    parts.append(f'object recorder {{\n    parent {target_node};\n    file "{OUTPUT_CSV.as_posix()}";\n    interval 900;\n    property voltage_A, voltage_B, voltage_C;\n}};\n')
    return '\n'.join(parts)


net_busphase, gld_mapping_summary, gld_npts = build_busphase_players()
glm_text = build_glm(net_busphase, gld_npts)
GLM_FILE.write_text(glm_text, encoding="utf-8")

GRIDLABD_EXE = Path(r"C:\Program Files\GridLAB-D 5.3.0\bin\gridlabd.exe")
_result = subprocess.run(
    [str(GRIDLABD_EXE), str(GLM_FILE)],
    cwd=str(BASE_DIR),
    capture_output=True,
    text=True,
)
if _result.returncode != 0:
    raise RuntimeError(
        "GridLAB-D simulation failed.\n" + _result.stdout[-4000:] + "\n" + _result.stderr[-8000:]
    )

_gld = pd.read_csv(OUTPUT_CSV, comment="#")
_gld.rename(columns={_gld.columns[0]: "timestamp"}, inplace=True)
_gld["timestamp"] = (
    _gld["timestamp"].astype(str)
    .str.replace(" EST", "", regex=False)
    .str.replace(" EDT", "", regex=False)
)
_gld["timestamp"] = pd.to_datetime(_gld["timestamp"], errors="coerce")

def _complex_mag(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("V", "").replace("d", "j").strip()
    try:
        return abs(complex(s))
    except Exception:
        return np.nan

VBASE_LN = 4160.0 / np.sqrt(3)
gld_vpu = {}
for col in _gld.columns[1:]:
    gld_vpu[col] = _gld[col].map(_complex_mag).to_numpy(float) / VBASE_LN

_a_col = next((c for c in _gld.columns[1:] if "voltage_A" in c), _gld.columns[1])
gld_vpu_63a = gld_vpu[_a_col]



# Build the exact 480-point comparison dataset used by the test notebook
base = np.asarray(sel_s3["63.1"], dtype=float)
ods = np.asarray(sel_s4["63.1"], dtype=float)
gld = np.asarray(gld_vpu_63a, dtype=float)

assert len(base) == len(ods) == len(gld), (
    f"Length mismatch: S3={len(base)}, OpenDSS S4={len(ods)}, GridLAB-D S4={len(gld)}"
)

open_dss_time = pd.date_range(
    start=pd.Timestamp(START_DATE) + pd.Timedelta(minutes=15),
    periods=len(ods),
    freq=FREQ,
)
gld_time = pd.to_datetime(_gld["timestamp"], errors="coerce")
assert len(gld_time) == len(gld)

df_ods = pd.DataFrame({
    "Timestamp": open_dss_time,
    "OpenDSS_S3_Vpu": base,
    "OpenDSS_S4_Vpu": ods,
})
df_gld = pd.DataFrame({
    "Timestamp": gld_time.to_numpy(),
    "GridLABD_S4_Vpu": gld,
})
comparison_df = pd.merge(df_ods, df_gld, on="Timestamp", how="inner")
comparison_df = (
    comparison_df.replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["OpenDSS_S3_Vpu", "OpenDSS_S4_Vpu", "GridLABD_S4_Vpu"])
    .reset_index(drop=True)
)
N = len(comparison_df)
assert N == len(base) == len(ods) == len(gld), (
    f"Timestamp alignment failed: S3={len(base)}, OpenDSS S4={len(ods)}, GridLAB-D S4={len(gld)}, paired={N}"
)
comparison_df["Absolute_Error_pu"] = np.abs(
    comparison_df["OpenDSS_S4_Vpu"] - comparison_df["GridLABD_S4_Vpu"]
)
mae = comparison_df["Absolute_Error_pu"].mean()
time_hours = np.arange(1, N + 1) * 0.25

