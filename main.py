#!/usr/bin/env python3
"""CLI entry point for the optimizer comparison framework.

Usage examples:

  # Regression (mock run, no GPU)
  python main.py --task-type regression --datasets superconductivity \
      --model-types simple_mlp --mock-run --no-ngrok

  # Full run on 4 GPUs
  python main.py --task-type all --gpu-num 4 --num-samples 40 \
      --output-dir /workspace/outputs --mlflow-uri /workspace/mlruns \
      --data-dir /workspace/data --no-ngrok

  # Image classification, specific datasets
  python main.py --task-type image_classification \
      --datasets places365 intel --model-types resnet18 efficientnet_v2_s \
      --num-epochs 10 --gpu-num 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    # Suppress noisy third-party loggers
    for noisy in ("ray", "ray.tune", "ray.air", "pytorch_lightning",
                  "lightning", "absl", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimizer comparison framework – trains all configured "
                    "optimizers on the given task and dataset(s) using Ray Tune.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task
    p.add_argument(
        "--task-type",
        choices=["regression", "tabular_classification", "image_classification", "all"],
        default="regression",
        help="Which task type to run. 'all' runs all three sequentially.",
    )

    # Dataset / model selection
    p.add_argument(
        "--datasets", nargs="+", default=[],
        help="Dataset name(s) to include. Defaults depend on task-type: "
             "regression→[superconductivity, yearmsd]; "
             "tabular_classification→[adult, creditcard]; "
             "image_classification→[places365, intel].",
    )
    p.add_argument(
        "--model-types", nargs="+", default=[],
        help="Model architecture(s). Defaults per task: "
             "regression→[simple_mlp, attention_mlp]; "
             "tabular_classification→[simple_cls, attention_cls]; "
             "image_classification→[resnet18, efficientnet_v2_s].",
    )

    # HPO
    p.add_argument("--num-samples", type=int, default=40,
                   help="Number of Optuna trials per optimizer.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Random seeds for multi-seed evaluation.")

    # Training
    p.add_argument("--batch-size", type=int, default=0,
                   help="Batch size (0 = task default: reg=256, cls=1024, img=64).")
    p.add_argument("--num-epochs", type=int, default=0,
                   help="Max training epochs (0 = task default: reg=30, cls=26, img=10).")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes.")

    # Hardware
    p.add_argument("--gpu-num", type=int, default=1,
                   help="Number of GPUs to use (0 = CPU only).")

    # Paths
    p.add_argument("--output-dir", default="./outputs",
                   help="Root directory for plots, CSVs, JSON history, telemetry.")
    p.add_argument("--mlflow-uri", default="./mlruns",
                   help="MLflow tracking URI (directory or sqlite:// URL).")
    p.add_argument("--data-dir", default="./data",
                   help="Dataset cache directory.")
    p.add_argument("--kaggle-json", default=None,
                   help="Path to kaggle.json credentials (for intel dataset download).")

    # MLflow UI exposure via ngrok
    p.add_argument("--ngrok-token", default=os.environ.get("NGROK_AUTH_TOKEN", ""),
                   help="ngrok auth token for MLflow UI tunnel.")
    p.add_argument("--no-ngrok", action="store_true",
                   help="Disable ngrok tunnel (required in Docker/cluster environments).")

    # Mock run
    p.add_argument("--mock-run", action="store_true",
                   help="Quick smoke-test: 4 HPO trials, 2 epochs, subsample data.")

    return p.parse_args()


def _start_mlflow_ui(mlflow_uri: str, ngrok_token: str) -> None:
    """Start MLflow UI in background and expose via ngrok."""
    import subprocess
    subprocess.Popen(
        ["mlflow", "ui", "--backend-store-uri", mlflow_uri, "--host", "0.0.0.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        from pyngrok import ngrok
        ngrok.kill()
        ngrok.set_auth_token(ngrok_token)
        url = ngrok.connect(5000, "http", host_header="localhost:5000")
        logging.getLogger(__name__).info("MLflow UI: %s", url)
    except Exception as e:
        logging.getLogger(__name__).warning("ngrok failed: %s", e)


def _run_task(task_type: str, args: argparse.Namespace) -> None:
    from src.runner import ExperimentConfig, TASK_DEFAULTS, run_experiments

    defaults = TASK_DEFAULTS.get(task_type, {})
    batch_size = args.batch_size or defaults.get("batch_size", 256)
    num_epochs = args.num_epochs or defaults.get("num_epochs", 30)

    cfg = ExperimentConfig(
        task_type=task_type,
        datasets=args.datasets or [],
        model_types=args.model_types or [],
        num_samples=args.num_samples,
        seeds=args.seeds,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gpu_num=args.gpu_num,
        mock_run=args.mock_run,
        output_dir=args.output_dir,
        mlflow_uri=args.mlflow_uri,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        kaggle_json=args.kaggle_json,
    )

    log = logging.getLogger(__name__)
    log.info("Starting task_type=%s  datasets=%s  models=%s",
             task_type, cfg.datasets, cfg.model_types)
    log.info("  gpu_num=%d  num_samples=%d  epochs=%d  mock=%s",
             cfg.gpu_num, cfg.num_samples, cfg.num_epochs, cfg.mock_run)

    run_experiments(cfg)


def main() -> None:
    args = _parse_args()
    _setup_logging(args.output_dir)
    log = logging.getLogger(__name__)

    if not args.no_ngrok and args.ngrok_token:
        _start_mlflow_ui(args.mlflow_uri, args.ngrok_token)

    task_types = (
        ["regression", "tabular_classification", "image_classification"]
        if args.task_type == "all"
        else [args.task_type]
    )

    for task_type in task_types:
        # For 'all', datasets/model_types are cleared so each task uses its defaults
        if args.task_type == "all":
            args.datasets = []
            args.model_types = []
        try:
            _run_task(task_type, args)
        except Exception as e:
            log.error("Task %s failed: %s", task_type, e, exc_info=True)
            if args.task_type != "all":
                sys.exit(1)

    log.info("All tasks completed. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
