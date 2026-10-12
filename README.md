## 1. Prerequisites

- Install Python 3.10.18
- Create a virtual environment: `py -3.10 -m venv .cahg`
- Activate the virtual environment: `.cahg\Scripts\activate.bat`
- Install Requirements:
  `python -m pip install --upgrade pip`
  `python -m pip install -r requirements.txt`
- Connect Jupyter Kernel:
  `python -m pip install ipykernel`
- OpenDSS Installation: v10.2.0.1 (64-bit)
- GridLab-D Installation: v5.3.0

## 2. File Names and Folder Structure

project/
├── README.md                  # Project overview and instructions
├── requirements.txt           # Python dependencies
├── Main.py                    # Python script for training and evaluation
├── Figures_Tables.ipynb       # Results analysis and visualization
├── agent_env.py               # EV environment, dynamics, and rewards
├── agent_policy.py            # Multi-agent RL algorithms and policies
├── gen_ev_model.py            # EV fleet and mobility generation
├── investigate_Vpu.py         # OpenDSS voltage analysis
├── vpp_forecast_pipeline.py   # VPP request forecasting pipeline using XGBoost
├── OpenDSS_GridLabD.py        # Power grid simulation execution
├── master_with_ev_pv_v2g.dss  # OpenDSS network model
├── csv_inputs/                # Input datasets
│   ├── v2ghome_bus.csv        # EV-to-bus mapping
│   ├── price_t.csv            # Electricity price data
│   └── ...                    # Load and generation profiles
├── vpp_forecast_output/       # Generated VPP forecast data
└── results/                   # Training and evaluation outputs

## 3. Execution and Results

- Execution: `Main.py`
- Results: `Figures_Tables.ipynb`