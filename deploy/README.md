# Deployment on vast.ai

## Prerequisites

```bash
pip install vastai
vastai set api-key <your_api_key>          # from console.vast.ai → Account
export DOCKERHUB_USER=yourdockerhubusername
docker login
```

## Step-by-step

### 1. Build & push Docker image

```bash
./deploy/build_and_push.sh              # pushes :latest
./deploy/build_and_push.sh v1.2         # pushes :v1.2
```

The image is based on `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` and
includes all Python dependencies plus the AdEMAMix git repo. Stacey++ is
vendored locally in `src/optimizers.py`.

### 2. Rent a GPU instance

```bash
# 4× RTX 4080 (16 GB VRAM, Ada sm_89) — recommended
./deploy/create_instance.sh RTX_4080 4 100

# 4× RTX 3090 (24 GB VRAM, Ampere sm_86)
./deploy/create_instance.sh RTX_3090 4 100

# 4× RTX 4090, 100 GB disk
./deploy/create_instance.sh RTX_4090 4 100

# 4× A100 80 GB, 200 GB disk
./deploy/create_instance.sh A100_SXM4_80GB 4 200
```

The instance ID is written to `.vastai_instance_id`.  
The `onstart.sh` script is automatically executed on boot, running:
```
python main.py --task-type all --gpu-num <N> --no-ngrok ...
```

### 3. Watch the run

```bash
./deploy/watch_instance.sh              # uses .vastai_instance_id
./deploy/watch_instance.sh <id>         # explicit ID
```

Or SSH directly:
```bash
vastai ssh-url <instance_id>
ssh <url from above>
tail -f /workspace/outputs/run.log
```

### 4. Pull results to local machine

```bash
./deploy/sync_results.sh                # uses .vastai_instance_id
./deploy/sync_results.sh <id> ./my_run  # save to custom dir
```

This rsyncs:
- `/workspace/outputs/` → `./outputs/`  (plots, CSVs, JSON, telemetry, run.log)
- `/workspace/mlruns/`  → `./mlruns/`   (MLflow tracking data)

### 5. Destroy instance (stop billing)

```bash
./deploy/destroy_instance.sh            # syncs first, then destroys
```

---

## Output structure (after sync)

```
outputs/
├── run.log                          # full stdout/stderr
├── best_run_plots/                  # HPO best-trial validation curves
├── tuned_run_plots/                 # Tuned multi-seed curves + resource bars
├── default_run_plots/               # Default-hparam curves + resource bars
├── varying_rs_run_plots/            # Seed-variability boxplots
└── results/
    ├── {dataset}_{model}_comparison.csv
    ├── {dataset}_{model}_tuned_summary.csv
    ├── {dataset}_{model}_default_summary.csv
    ├── {dataset}_{model}_rs_results.csv
    ├── {dataset}_{model}_run_history.json
    ├── cross_summary.csv
    └── telemetry/
        └── {dataset}_{model}_telemetry.jsonl
```

## Viewing MLflow results locally

```bash
mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000
```

## Analysing telemetry

```python
from src.telemetry import load_telemetry, telemetry_summary

df = load_telemetry("outputs/results/telemetry/superconductivity_simple_mlp_telemetry.jsonl")
print(telemetry_summary(df))
```

## Custom run (not full 'all')

Edit `deploy/onstart.sh`, or override via `vastai create instance`:
```bash
vastai create instance <offer_id> \
    --image "$DOCKERHUB_USER/optimizer-runner:latest" \
    --disk 100 \
    --onstart "cd /workspace && python main.py --task-type regression \
        --datasets superconductivity --model-types simple_mlp attention_mlp \
        --gpu-num 4 --num-samples 40 --no-ngrok \
        2>&1 | tee outputs/run.log"
```
