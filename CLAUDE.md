# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Framework for comparing neural network optimizers (Lion, AdEMAMix, AdamWScheduleFree, GaLoreAdamW, Stacey_p2, AdamW, AdaHessian, AdaBelief, AdaBound) across three task types:

- **Regression** – Superconductivity, YearPredictionMSD (UCI download)
- **Tabular classification** – Adult, CreditCard, Higgs-small, Jannis, Helena (OpenML)
- **Image classification** – Places365 (torchvision), Intel image (Kaggle/ImageFolder)

Tracking & HPO: MLflow + Ray Tune (OptunaSearch + ASHAScheduler).  
Target deployment: **vast.ai GPU cluster** (4+ modern GPUs, Ampere/Ada/Hopper).

## Setup & Running

```bash
pip install -r requirements.txt
# Clone optimizer repos not on PyPI:
git clone https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch AdEMAMix_Optimizer_Pytorch
git clone https://github.com/xinyuluo8561/Stacey
```

Primary interaction is through the CLI (`main.py`) or Jupyter notebooks.

## CLI Reference

```bash
python main.py \
  --task-type   regression|tabular_classification|image_classification|all \
  --datasets    superconductivity yearmsd \   # override defaults
  --model-types simple_mlp attention_mlp \   # override defaults
  --num-samples 60 \                          # Optuna trials per optimizer
  --seeds       0 1 2 3 4 \
  --batch-size  256 \                         # 0 = task default
  --num-epochs  30 \                          # 0 = task default
  --gpu-num     4 \                           # 0 = CPU only
  --output-dir  ./outputs \
  --mlflow-uri  ./mlruns \
  --data-dir    ./data \                      # dataset cache
  --kaggle-json ~/.kaggle/kaggle.json \       # for intel dataset download
  --no-ngrok \                                # skip MLflow UI tunnel
  --mock-run                                  # smoke test: 4 trials, 2 epochs
```

### Task defaults

| task-type | datasets | model-types | batch | epochs |
|---|---|---|---|---|
| regression | superconductivity, yearmsd | simple_mlp, attention_mlp | 256 | 30 |
| tabular_classification | adult, creditcard | simple_cls, attention_cls | 1024 | 26 |
| image_classification | places365, intel | resnet18, efficientnet_v2_s | 64 | 10 |

### Mock run (no GPU, quick smoke test)

```bash
python main.py --task-type regression --datasets superconductivity \
    --model-types simple_mlp --mock-run --no-ngrok --gpu-num 0
```

## Output Directory Structure

```
outputs/
├── run.log                          # full stdout/stderr
├── best_run_plots/                  # HPO best-trial validation curves (PNG)
├── tuned_run_plots/                 # Tuned multi-seed val curves + resource bars
├── default_run_plots/               # Default-hparam curves + resource bars
├── varying_rs_run_plots/            # Seed-variability boxplots
└── results/
    ├── {exp_key}_comparison.csv     # HPO best configs per optimizer
    ├── {exp_key}_tuned_summary.csv  # Tuned multi-seed stats
    ├── {exp_key}_default_summary.csv
    ├── {exp_key}_rs_results.csv     # Per-seed test results
    ├── {exp_key}_run_history.json   # Full seed-level history
    ├── cross_summary.csv            # Cross-experiment pivot table
    └── telemetry/
        └── {exp_key}_telemetry.jsonl  # Per-run CPU/RAM/GPU JSONL
```

`exp_key` = `{dataset}_{model_type}`, e.g. `superconductivity_simple_mlp`.

### Viewing MLflow results locally

```bash
mlflow ui --backend-store-uri ./mlruns   # http://localhost:5000
```

### Analysing telemetry

```python
from src.telemetry import load_telemetry, telemetry_summary
df = load_telemetry("outputs/results/telemetry/superconductivity_simple_mlp_telemetry.jsonl")
print(telemetry_summary(df))
```

## vast.ai Deployment

See `deploy/README.md` for full instructions. Quick summary:

```bash
# 1. Build & push image (once, or on code changes)
export DOCKERHUB_USER=yourusername
./deploy/build_and_push.sh

# 2. Rent instance and start run
./deploy/create_instance.sh RTX_4090 4 100

# 3. Monitor live
./deploy/watch_instance.sh          # uses .vastai_instance_id

# 4. Pull results
./deploy/sync_results.sh

# 5. Stop billing
./deploy/destroy_instance.sh
```

Volumes (persist across container restarts):
- `/workspace/outputs` → local `./outputs`
- `/workspace/mlruns`  → local `./mlruns`
- `/workspace/data`    → local `./data` (dataset cache, downloaded once)

## Source Layout

```
src/
├── config.py          # Seeds, optimizer grids (Optuna loguniform), detect_gpu_resources()
├── datasets.py        # Unified registry + task-specific loaders with download
├── downloader.py      # download_zip_member(), resolve_local_path(), download_kaggle_dataset()
├── model_builder.py   # MLP, SimpleMLP, AttentionMLP, TabularAttentionModel, build_tabular_model()
├── vision_models.py   # resnet18, efficientnet_v2_s, small_cnn, build_vision_model()
├── lightning_wrapper.py # LightningWrapper (cls), RegressionLightningWrapper
├── callbacks.py       # ScheduleFreeOptimizerCallback, SystemMonitorCallback (CPU/RAM/GPU)
├── telemetry.py       # TelemetryLogger (JSONL), load_telemetry(), telemetry_summary()
├── runner.py          # ExperimentConfig, run_experiments() — full 6-step loop
├── stats.py           # aggregate_seeds(), best_config_per_scenario(), cross_experiment_summary()
└── optimizers.py      # AdaBound, GaLoreAdamW wrapper
```

## Rules

- Ask follow-up questions before doing complex tasks until you reach 95% confidence.
- Datasets not in sklearn/torchvision must use `src/downloader.py` helpers with local cache first.
- All plots saved to files (matplotlib Agg backend); never call `plt.show()` in non-notebook code.
- Resource monitoring is done by `SystemMonitorCallback`; telemetry written by `TelemetryLogger`.
- Docker image must not bake in `outputs/`, `mlruns/`, or `data/` — those come from volumes.
