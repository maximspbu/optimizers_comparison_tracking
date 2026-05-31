"""Full 6-step experiment runner mirroring the three Jupyter notebooks.

Steps per (dataset, model_type) pair:
  1. HPO  – Ray Tune + OptunaSearch + ASHA + MLflowLoggerCallback
  2. Comparison DataFrame  – best config per optimizer
  3. Best-trial val metric plots  → output_dir/best_run_plots/
  4. Tuned multi-seed evaluation  → output_dir/tuned_run_plots/
  5. Default (no-tune) evaluation → output_dir/default_run_plots/
  6. Varying-seed boxplots        → output_dir/varying_rs_run_plots/

All results are also written to output_dir/results/ as CSV / JSON / JSONL.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ray
import seaborn as sns
import torch
import ray.air as _ray_air
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from . import config as cfg
from .callbacks import ScheduleFreeOptimizerCallback, SystemMonitorCallback
from .datasets import (
    REGRESSION_DATASETS,
    TABULAR_CLS_DATASETS,
    IMAGE_CLS_DATASETS,
    load_tabular_regression,
    load_openml_tabular,
    load_image_dataset,
)
from .lightning_wrapper import LightningWrapper, RegressionLightningWrapper
from .model_builder import build_tabular_model, list_tabular_models
from .telemetry import TelemetryLogger
from .vision_models import build_vision_model, list_vision_models

log = logging.getLogger(__name__)

sns.set_style("whitegrid")


def _make_reporter(frequency: int = 30) -> Any:
    """Return JupyterNotebookReporter in notebooks, CLIReporter otherwise."""
    try:
        from ray.tune.progress_reporter import JupyterNotebookReporter
        return JupyterNotebookReporter(overwrite=True, max_report_frequency=frequency)
    except Exception:
        pass
    return CLIReporter(max_report_frequency=frequency)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

TASK_DEFAULTS: dict[str, dict] = {
    "regression": {
        "datasets": list(REGRESSION_DATASETS),
        "model_types": ["simple_mlp", "attention_mlp"],
        "batch_size": 256,
        "num_epochs": 100,
    },
    "tabular_classification": {
        "datasets": list(TABULAR_CLS_DATASETS),
        "model_types": ["simple_cls", "attention_cls"],
        "batch_size": 1024,
        "num_epochs": 100,
    },
    "image_classification": {
        "datasets": list(IMAGE_CLS_DATASETS),
        "model_types": ["resnet18", "efficientnet_v2_s"],
        "batch_size": 64,
        "num_epochs": 50,
    },
}

# Early stopping patience per task — how many val epochs without improvement before stop
_ES_PATIENCE: dict[str, int] = {
    "regression": 20,
    "tabular_classification": 15,
    "image_classification": 10,
}


@dataclass
class ExperimentConfig:
    task_type: str                                    # regression | tabular_classification | image_classification
    datasets: list[str] = field(default_factory=list)
    model_types: list[str] = field(default_factory=list)
    num_samples: int = 96
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    batch_size: int = 256
    num_epochs: int = 30
    gpu_num: int = 1
    mock_run: bool = False
    mock_max_rows: int = 1000
    mock_max_imgs: int = 200
    output_dir: str = "./outputs"
    mlflow_uri: str = "./mlruns"
    data_dir: str = "./data"
    num_workers: int = 4
    kaggle_json: str | None = None

    def __post_init__(self) -> None:
        defaults = TASK_DEFAULTS.get(self.task_type, {})
        if not self.datasets:
            self.datasets = defaults.get("datasets", [])
        if not self.model_types:
            self.model_types = defaults.get("model_types", [])
        if self.mock_run:
            self.num_samples = 4
            self.num_epochs = 2
            self.seeds = self.seeds[:1]


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _makedirs(output_dir: str) -> None:
    for subdir in (
        "best_run_plots", "tuned_run_plots", "default_run_plots",
        "varying_rs_run_plots", "results", "results/telemetry",
    ):
        Path(output_dir, subdir).mkdir(parents=True, exist_ok=True)


def _plot_path(output_dir: str, subdir: str, exp_key: str, suffix: str) -> str:
    return str(Path(output_dir, subdir, f"{exp_key}_{suffix}.png"))


def _result_path(output_dir: str, exp_key: str, suffix: str) -> str:
    return str(Path(output_dir, "results", f"{exp_key}_{suffix}"))


def _telemetry_path(output_dir: str, exp_key: str) -> str:
    return str(Path(output_dir, "results", "telemetry", f"{exp_key}_telemetry.jsonl"))


# ---------------------------------------------------------------------------
# Dataset loading per task type
# ---------------------------------------------------------------------------

def _load_datasets(
    task_type: str,
    dataset_name: str,
    cfg_obj: ExperimentConfig,
) -> tuple[Any, Any, Any, int, int]:
    """Return (train_ds, val_ds, test_ds, n_features, n_classes)."""
    max_rows = cfg_obj.mock_max_rows if cfg_obj.mock_run else None
    seed = cfg_obj.seeds[0]

    if task_type == "regression":
        train_ds, val_ds, test_ds, n_feat, _, _ = load_tabular_regression(
            dataset_name, seed=seed, data_dir=cfg_obj.data_dir,
            max_rows=max_rows,
        )
        return train_ds, val_ds, test_ds, n_feat, 0

    if task_type == "tabular_classification":
        train_ds, val_ds, test_ds, n_feat, n_cls = load_openml_tabular(
            dataset_name, seed=seed, data_dir=cfg_obj.data_dir,
        )
        if max_rows:
            from torch.utils.data import Subset as _Subset
            import torch as _torch
            def _sub(ds, n):
                idx = _torch.randperm(len(ds))[:n].tolist()
                return _Subset(ds, idx)
            train_ds = _sub(train_ds, int(max_rows * 0.70))
            val_ds   = _sub(val_ds,   int(max_rows * 0.15))
            test_ds  = _sub(test_ds,  int(max_rows * 0.15))
        return train_ds, val_ds, test_ds, n_feat, n_cls

    if task_type == "image_classification":
        train_ds, val_ds, test_ds, n_cls = load_image_dataset(
            dataset_name, seed=seed, data_dir=cfg_obj.data_dir,
            kaggle_json=cfg_obj.kaggle_json,
        )
        if max_rows:
            from torch.utils.data import Subset as _Subset
            import torch as _torch
            def _sub(ds, n):
                idx = _torch.randperm(len(ds))[:n].tolist()
                return _Subset(ds, idx)
            train_ds = _sub(train_ds, int(cfg_obj.mock_max_imgs * 0.70))
            val_ds   = _sub(val_ds,   int(cfg_obj.mock_max_imgs * 0.15))
            test_ds  = _sub(test_ds,  int(cfg_obj.mock_max_imgs * 0.15))
        return train_ds, val_ds, test_ds, -1, n_cls

    raise ValueError(f"unknown task_type: {task_type!r}")


# ---------------------------------------------------------------------------
# Model / wrapper builders
# ---------------------------------------------------------------------------

def _build_model(
    task_type: str, model_type: str, n_features: int, n_classes: int
) -> nn.Module:
    hidden = max(64, n_features) if n_features > 0 else 64
    if task_type == "image_classification":
        return build_vision_model(model_type, num_classes=n_classes)
    if task_type == "regression":
        return build_tabular_model(model_type, input_shape=n_features, output_shape=1, hidden_units=hidden)
    return build_tabular_model(model_type, input_shape=n_features, output_shape=n_classes, hidden_units=hidden)


def _make_wrapper(
    task_type: str,
    model: nn.Module,
    optimizer_class,
    hparams: dict,
    n_classes: int,
):
    """Return (Lightning wrapper, tune_metrics dict)."""
    if task_type == "regression":
        wrapper = RegressionLightningWrapper(
            model=model, optimizer_class=optimizer_class,
            optimizer_hparams=hparams, loss_fn=nn.MSELoss(),
        )
        tune_metrics = {
            "val_loss": "val_loss", "val_mse": "val_mse",
            "val_mae": "val_mae", "val_r2": "val_r2", "val_rmse": "val_rmse",
        }
    else:
        wrapper = LightningWrapper(
            model=model, optimizer_class=optimizer_class,
            optimizer_hparams=hparams, loss_fn=nn.CrossEntropyLoss(),
            num_classes=n_classes,
        )
        tune_metrics = {
            "val_loss": "val_loss", "val_accuracy": "val_accuracy",
            "val_f1score": "val_f1score", "val_precision": "val_precision",
            "val_recall": "val_recall",
        }
    return wrapper, tune_metrics


def _hpo_metric(task_type: str) -> tuple[str, str]:
    if task_type == "regression":
        return "val_r2", "max"
    return "val_accuracy", "max"


def _bad_result(task_type: str) -> dict:
    if task_type == "regression":
        return {"val_loss": float("inf"), "val_mse": float("inf"),
                "val_mae": float("inf"), "val_r2": -float("inf"), "val_rmse": float("inf")}
    return {"val_loss": float("inf"), "val_accuracy": 0.0,
            "val_f1score": 0.0, "val_precision": 0.0, "val_recall": 0.0}


def _test_metric_keys(task_type: str) -> list[str]:
    if task_type == "regression":
        return ["test_mse", "test_mae", "test_r2", "test_rmse"]
    return ["test_accuracy", "test_f1score", "test_precision", "test_recall"]


# ---------------------------------------------------------------------------
# HPO
# ---------------------------------------------------------------------------

def _run_hpo(
    task_type: str,
    dataset_name: str,
    model_type: str,
    cfg_obj: ExperimentConfig,
    train_ref,
    val_ref,
    n_features: int,
    n_classes: int,
    optimizer_params: dict,
    exp_name: str,
    gpu_per_trial: float,
    cpu_per_trial: float,
    max_concurrent: int,
) -> dict:
    """Run HPO for all optimizers via tune.Tuner (Ray 2.x API).

    Returns {optimizer_class: ResultGrid}.

    Key design decision: ConcurrencyLimiter is intentionally NOT used.
    It tracks "outstanding suggestions" independently of Ray's execution
    scheduler and can refuse to yield new configs even when trial slots
    are free, leaving GPUs idle. max_concurrent_trials in TuneConfig is
    the authoritative concurrency control.
    """
    import optuna as _optuna

    metric, mode = _hpo_metric(task_type)
    skip_keys = cfg.SKIP_KEYS
    _NUM_FEATURES = n_features
    _N_CLASSES = n_classes
    _TASK = task_type
    _MODEL_TYPE = model_type
    _BATCH = cfg_obj.batch_size
    _EPOCHS = cfg_obj.num_epochs
    _CPU_PER_TRIAL = cpu_per_trial
    _es_patience = max(5, _ES_PATIENCE.get(task_type, 15))

    # n_startup_trials >= max_concurrent: first full batch uses pure random
    # sampling so all concurrent slots fill immediately on start.
    _n_startup = max(max_concurrent, cfg_obj.num_samples // 3)
    # Ray defaults TUNE_MAX_PENDING_TRIALS_PG=1 for non-Basic searchers.
    # OptunaSearch then creates only one pending placement group at a time,
    # so slow Lightning trial startup leaves most fractional-GPU slots idle.
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(max_concurrent)

    result_grids: dict = {}

    for optimizer_class, hparam_space in optimizer_params.items():
        opt_name = optimizer_class.__name__
        log.info("HPO  optimizer=%s  samples=%d  startup=%d  max_concurrent=%d",
                 opt_name, cfg_obj.num_samples, _n_startup, max_concurrent)

        _opt_cls = optimizer_class

        def wrap(config: dict) -> None:
            import sys, os as _os, warnings as _w, logging as _l
            import torch as _torch
            sys.stderr = open(_os.devnull, "w")
            _w.filterwarnings("ignore")
            _l.getLogger("pytorch_lightning").setLevel(_l.ERROR)
            sys.unraisablehook = lambda *_: None

            _n_torch_threads = max(1, int(_CPU_PER_TRIAL))
            _n_workers = (
                max(0, min(int(_CPU_PER_TRIAL) - 1, 4))
                if _TASK == "image_classification" else 0
            )
            _torch.set_num_threads(_n_torch_threads)

            import pytorch_lightning as _pl
            from ray.tune.integration.pytorch_lightning import TuneReportCallback
            from torch.utils.data import DataLoader as _DL
            import ray as _ray

            _pl.seed_everything(config.get("seed", 0), workers=True)
            train_ds = _ray.get(train_ref)
            val_ds   = _ray.get(val_ref)

            hparams = {k: v for k, v in config.items() if k not in skip_keys}
            if hasattr(_opt_cls, "__name__") and "Lion" in _opt_cls.__name__:
                hparams["use_triton"] = False
            if hasattr(_opt_cls, "__name__") and "GaLore" in _opt_cls.__name__:
                hparams["no_deprecation_warning"] = True

            model = _build_model(_TASK, _MODEL_TYPE, _NUM_FEATURES, _N_CLASSES)
            wrapper, tune_metrics = _make_wrapper(_TASK, model, _opt_cls, hparams, _N_CLASSES)
            tune_cb = TuneReportCallback(tune_metrics, on="validation_end")
            from .callbacks import ScheduleFreeOptimizerCallback as _SFCB
            from pytorch_lightning.callbacks import EarlyStopping as _ES
            sfcb = _SFCB()
            es = _ES(monitor="val_loss", patience=_es_patience, mode="min",
                     min_delta=1e-4, check_on_train_epoch_end=False)

            trainer = _pl.Trainer(
                callbacks=[tune_cb, sfcb, es],
                max_epochs=_EPOCHS,
                accelerator="auto", devices=1,
                precision="16-mixed" if _TASK != "regression" else "32-true",
                enable_progress_bar=False, enable_checkpointing=False,
                enable_model_summary=False,
                logger=False,
            )
            bad = _bad_result(_TASK)
            try:
                dl_kw = dict(batch_size=_BATCH, num_workers=_n_workers,
                             pin_memory=True, persistent_workers=_n_workers > 0)
                trainer.fit(wrapper, _DL(train_ds, shuffle=True, **dl_kw),
                            _DL(val_ds, shuffle=False, **dl_kw))
            except (AssertionError, RuntimeError) as e:
                if any(k in str(e).lower() for k in ("grad", "c_tmp", "nan")):
                    tune.report(bad); return
                raise
            except Exception as e:
                if "nan" in str(e).lower():
                    tune.report(bad); return
                raise

        sampler = _optuna.samplers.TPESampler(
            n_startup_trials=_n_startup,
            seed=42,
            multivariate=True,
            group=True,
            constant_liar=True,  # pending trials treated as pessimistic lies → full parallel throughput
        )
        searcher = OptunaSearch(sampler=sampler, metric=metric, mode=mode)

        mlflow_cb = MLflowLoggerCallback(
            tracking_uri=cfg_obj.mlflow_uri,
            experiment_name=f"{exp_name}_{opt_name}_tune",
            save_artifact=False,
        )

        trainable = tune.with_resources(
            wrap, resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial}
        )
        tuner = tune.Tuner(
            trainable,
            param_space=hparam_space | {"seed": tune.choice(cfg_obj.seeds)},
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                search_alg=searcher,
                scheduler=ASHAScheduler(
                    max_t=cfg_obj.num_epochs,
                    grace_period=max(1, cfg_obj.num_epochs // 4),
                    reduction_factor=2, brackets=1,
                ),
                num_samples=cfg_obj.num_samples,
                max_concurrent_trials=max_concurrent,
            ),
            run_config=_ray_air.RunConfig(
                storage_path=str(Path(cfg_obj.output_dir).resolve() / "ray_results"),
                log_to_file=True,
                callbacks=[mlflow_cb],
                progress_reporter=_make_reporter(frequency=30),
                verbose=1,
            ),
        )
        result_grid = tuner.fit()
        result_grids[optimizer_class] = result_grid

        best = result_grid.get_best_result(metric=metric, mode=mode)
        if best:
            val = (best.metrics or {}).get(metric, float("nan"))
            log.info("  best %s = %.4f", metric, val)

    return result_grids


# ---------------------------------------------------------------------------
# Evaluation (tuned / default / varying-seeds)
# ---------------------------------------------------------------------------

def _run_evaluation(
    task_type: str,
    dataset_name: str,
    model_type: str,
    cfg_obj: ExperimentConfig,
    train_ds,
    val_ds,
    test_ds,
    n_features: int,
    n_classes: int,
    optimizers_with_configs: list[tuple],  # [(opt_class, hparams_dict), ...]
    exp_name: str,
    telemetry_logger: TelemetryLogger,
    exp_key: str,
) -> dict[str, list[dict]]:
    """Run multi-seed training for each optimizer and return {opt_name: [seed_result, ...]}."""
    mlflow.set_tracking_uri(cfg_obj.mlflow_uri)
    exp_id = cfg.get_or_create_experiment(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)

    hidden = max(64, n_features) if n_features > 0 else 64
    dl_kw = dict(
        batch_size=cfg_obj.batch_size,
        num_workers=cfg_obj.num_workers,
        pin_memory=True,
        persistent_workers=cfg_obj.num_workers > 0,
    )
    train_dl = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_dl  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    seed_results: dict[str, list[dict]] = {}

    for optimizer_class, hparams in optimizers_with_configs:
        opt_name = optimizer_class.__name__
        seed_results[opt_name] = []

        for seed in cfg_obj.seeds:
            pl.seed_everything(seed, workers=True)
            with mlflow.start_run(run_name=f"{exp_name}_{opt_name}_seed_{seed}"):
                mlflow.log_param("seed", seed)
                mlflow.log_param("optimizer", opt_name)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("dataset", dataset_name)

                model = _build_model(task_type, model_type, n_features, n_classes)
                wrapper, _ = _make_wrapper(task_type, model, optimizer_class, hparams, n_classes)
                mlflow.pytorch.autolog(log_every_n_step=0)
                sys_mon = SystemMonitorCallback()
                sfcb = ScheduleFreeOptimizerCallback()
                es = EarlyStopping(
                    monitor="val_loss",
                    patience=_ES_PATIENCE.get(task_type, 15),
                    mode="min", min_delta=1e-4,
                    check_on_train_epoch_end=False,
                )
                tb_logger = TensorBoardLogger(
                    save_dir=str(Path(cfg_obj.output_dir) / "tensorboard"),
                    name=exp_key,
                    version=f"{opt_name}_seed{seed}",
                )
                trainer = pl.Trainer(
                    callbacks=[sfcb, sys_mon, es],
                    logger=tb_logger,
                    max_epochs=cfg_obj.num_epochs,
                    accelerator="auto",
                    devices=max(1, cfg_obj.gpu_num),
                    strategy="ddp" if cfg_obj.gpu_num > 1 else "auto",
                    precision="16-mixed" if task_type != "regression" else "32-true",
                    enable_progress_bar=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                )
                test_keys = _test_metric_keys(task_type)
                try:
                    start = time.perf_counter()
                    trainer.fit(wrapper, train_dl, val_dl)
                    elapsed = time.perf_counter() - start
                    m = trainer.test(wrapper, test_dl)[0]
                    ms = {k: m.get(k, float("nan")) for k in test_keys}
                    ms["time_training"] = elapsed
                except (AssertionError, RuntimeError) as e:
                    if any(k in str(e).lower() for k in ("grad", "c_tmp", "nan")):
                        log.warning("NaN instability: %s seed=%d", opt_name, seed)
                        ms = {k: float("nan") for k in test_keys}
                        ms["time_training"] = float("inf")
                    else:
                        raise
                mon = sys_mon.summary()
                ms.update(mon)
                ms["seed"] = seed
                mlflow.log_metrics({k: v for k, v in ms.items() if isinstance(v, (int, float))})
                telemetry_logger.log_summary(exp_key, opt_name, seed, mon)

            log.info("  %s seed=%d  %s", opt_name, seed,
                     "  ".join(f"{k}={v:.4f}" for k, v in ms.items()
                                if k in test_keys and isinstance(v, float)))
            seed_results[opt_name].append(ms)

    return seed_results


# ---------------------------------------------------------------------------
# Summary DataFrames
# ---------------------------------------------------------------------------

def _build_tuned_summary(seed_results: dict[str, list[dict]], task_type: str) -> pd.DataFrame:
    rows = []
    for opt_name, runs in seed_results.items():
        if task_type == "regression":
            r2s   = [r["test_r2"]   for r in runs if not np.isnan(r.get("test_r2",   float("nan")))]
            rmses = [r["test_rmse"] for r in runs if not np.isnan(r.get("test_rmse", float("nan")))]
            rows.append({
                "optimizer": opt_name,
                "r2_median":   statistics.median(r2s)   if r2s   else float("nan"),
                "r2_max":      max(r2s)                 if r2s   else float("nan"),
                "rmse_median": statistics.median(rmses) if rmses else float("nan"),
                "rmse_min":    min(rmses)               if rmses else float("nan"),
                "n_seeds": len(runs),
            })
        else:
            accs = [r["test_accuracy"] for r in runs if not np.isnan(r.get("test_accuracy", float("nan")))]
            rows.append({
                "optimizer": opt_name,
                "acc_min":    min(accs)                 if accs else float("nan"),
                "acc_median": statistics.median(accs)   if accs else float("nan"),
                "acc_max":    max(accs)                 if accs else float("nan"),
                "acc_std":    statistics.stdev(accs)    if len(accs) > 1 else 0.0,
                "n_seeds": len(runs),
            })
    return pd.DataFrame(rows).set_index("optimizer")


def _build_comparison_df(
    task_type: str,
    analysis_results: dict,
) -> pd.DataFrame:
    metric, mode = _hpo_metric(task_type)
    rows = []
    for opt_cls, result_grid in analysis_results.items():
        best = result_grid.get_best_result(metric=metric, mode=mode)
        if best is None:
            continue
        best_cfg = best.config or {}
        res = best.metrics or {}
        row: dict = {
            "Optimizer": opt_cls.__name__,
            "Config": best_cfg,
            metric.replace("val_", "").upper(): res.get(metric, float("nan")),
            "Time_total_s": round(res.get("time_total_s", float("nan")), 2),
        }
        if task_type == "regression":
            row["RMSE"] = res.get("val_rmse", float("nan"))
            row["MAE"]  = res.get("val_mae", float("nan"))
        else:
            row["F1Score"]   = res.get("val_f1score", float("nan"))
            row["Precision"] = res.get("val_precision", float("nan"))
            row["Recall"]    = res.get("val_recall", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_metric_curves(
    analysis_results: dict,
    metric_name: str,
    title: str,
    out_path: str,
) -> None:
    """Plot per-epoch validation metric for the best trial of each optimizer."""
    mode = "max" if ("r2" in metric_name or "accuracy" in metric_name) else "min"
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for opt_cls, result_grid in analysis_results.items():
        try:
            best = result_grid.get_best_result(metric=metric_name, mode=mode)
            if best is None:
                continue
            # Try per-epoch curve from metrics_dataframe
            mdf = best.metrics_dataframe
            if mdf is not None and metric_name in mdf.columns:
                ax.plot(mdf[metric_name].values,
                        label=f"{opt_cls.__name__} (final={mdf[metric_name].iloc[-1]:.4f})")
            else:
                # Fallback: single final-value horizontal line
                val = (best.metrics or {}).get(metric_name, float("nan"))
                ax.axhline(val, linestyle="--",
                           label=f"{opt_cls.__name__} (final={val:.4f})", alpha=0.8)
            plotted = True
        except Exception as e:
            log.debug("Could not plot %s for %s: %s", metric_name, opt_cls.__name__, e)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    if plotted:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved: %s", out_path)


def _save_resource_bars(
    results: dict[str, dict],
    title_suffix: str,
    output_dir: str,
    exp_key: str,
    plot_subdir: str,
) -> None:
    rows = [
        {
            "optimizer": k,
            "avg_cpu_pct": v.get("avg_cpu_pct", 0),
            "avg_ram_gb":  v.get("avg_ram_gb", 0),
            "avg_gpu_gb":  v.get("avg_gpu_gb", 0),
        }
        for k, v in results.items()
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label in zip(
        axes,
        ["avg_cpu_pct", "avg_ram_gb", "avg_gpu_gb"],
        ["CPU %", "RAM GB", "GPU Memory GB"],
    ):
        sns.barplot(data=df, x="optimizer", y=col, ax=ax)
        ax.set_title(f"{label}  {title_suffix}")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = _plot_path(output_dir, plot_subdir, exp_key, "resources")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path)


def _save_boxplots(
    seed_results: dict[str, list[dict]],
    task_type: str,
    output_dir: str,
    exp_key: str,
) -> None:
    rows = [
        {**r, "optimizer": opt_name}
        for opt_name, runs in seed_results.items()
        for r in runs
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    if task_type == "regression":
        metrics_to_plot = ["test_r2", "test_rmse", "test_mae"]
    else:
        metrics_to_plot = ["test_accuracy", "test_f1score", "test_precision", "test_recall"]

    for m in metrics_to_plot:
        if m not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x="optimizer", y=m, ax=ax)
        ax.set_title(f"{m}  [{exp_key}]")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        path = _plot_path(output_dir, "varying_rs_run_plots", exp_key, m)
        plt.savefig(path, dpi=200)
        plt.close(fig)
        log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
    log.info("Saved: %s", path)


def _save_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Per-experiment orchestrator
# ---------------------------------------------------------------------------

def _run_experiment_pair(
    task_type: str,
    dataset_name: str,
    model_type: str,
    cfg_obj: ExperimentConfig,
    optimizer_params: dict,
    gpu_per_trial: float,
    cpu_per_trial: float,
    max_concurrent: int,
) -> dict:
    exp_key = f"{dataset_name}_{model_type}"
    log.info("\n%s\n  EXPERIMENT: dataset=%s  model=%s\n%s",
             "#" * 70, dataset_name, model_type, "#" * 70)

    # 0. Prepare output
    telemetry_logger = TelemetryLogger(_telemetry_path(cfg_obj.output_dir, exp_key))

    # 1. Load dataset
    log.info("Loading %s ...", dataset_name)
    train_ds, val_ds, test_ds, n_features, n_classes = _load_datasets(
        task_type, dataset_name, cfg_obj
    )
    log.info("  n_features=%d  n_classes=%d  train=%d  val=%d  test=%d",
             n_features, n_classes, len(train_ds), len(val_ds), len(test_ds))  # type: ignore[arg-type]

    # 2. HPO
    exp_name = f"{task_type[:3]}_{dataset_name}_{model_type}"
    train_ref = ray.put(train_ds)
    val_ref   = ray.put(val_ds)

    analysis_results = _run_hpo(
        task_type, dataset_name, model_type, cfg_obj,
        train_ref, val_ref, n_features, n_classes,
        optimizer_params, exp_name,
        gpu_per_trial, cpu_per_trial, max_concurrent,
    )

    # 3. Comparison DataFrame
    comparison_df = _build_comparison_df(task_type, analysis_results)
    _save_df(comparison_df, _result_path(cfg_obj.output_dir, exp_key, "comparison.csv"))

    # 3b. Best-trial metric curves (using best trial's final values as reference)
    metric, mode = _hpo_metric(task_type)
    for m in (["val_r2", "val_rmse"] if task_type == "regression" else ["val_accuracy", "val_f1score"]):
        _save_metric_curves(
            analysis_results, m,
            f"HPO best-trial {m}  [{exp_key}]",
            _plot_path(cfg_obj.output_dir, "best_run_plots", exp_key, m),
        )

    # Build (optimizer_class, best_config) pairs from ResultGrid objects
    tuned_pairs = []
    for opt_cls, result_grid in analysis_results.items():
        best = result_grid.get_best_result(metric=metric, mode=mode)
        best_cfg = best.config if best else {}
        hparams = {k: v for k, v in best_cfg.items() if k not in cfg.SKIP_KEYS}
        if "Lion" in opt_cls.__name__:
            hparams["use_triton"] = False
        if "GaLore" in opt_cls.__name__:
            hparams["no_deprecation_warning"] = True
        tuned_pairs.append((opt_cls, hparams))

    # 4. Tuned multi-seed evaluation
    log.info("--- Tuned multi-seed evaluation ---")
    tuned_exp = f"tuned_{exp_name}"
    tuned_seed_results = _run_evaluation(
        task_type, dataset_name, model_type, cfg_obj,
        train_ds, val_ds, test_ds, n_features, n_classes,
        tuned_pairs, tuned_exp, telemetry_logger, exp_key,
    )
    tuned_summary = _build_tuned_summary(tuned_seed_results, task_type)
    _save_df(tuned_summary, _result_path(cfg_obj.output_dir, exp_key, "tuned_summary.csv"))

    tuned_best = {
        opt_name: max(
            [r for r in runs if not np.isnan(r.get(
                "test_r2" if task_type == "regression" else "test_accuracy", float("nan")
            ))],
            key=lambda r: r.get("test_r2" if task_type == "regression" else "test_accuracy", float("-inf")),
            default=runs[0] if runs else {},
        )
        for opt_name, runs in tuned_seed_results.items()
    }
    _save_resource_bars(tuned_best, f"Tuned [{exp_key}]", cfg_obj.output_dir, exp_key, "tuned_run_plots")

    # 5. Default (no-tune) multi-seed evaluation
    log.info("--- Default multi-seed evaluation ---")
    default_pairs = [(opt_cls, {}) for opt_cls, _ in tuned_pairs]
    default_exp = f"default_{exp_name}"
    default_seed_results = _run_evaluation(
        task_type, dataset_name, model_type, cfg_obj,
        train_ds, val_ds, test_ds, n_features, n_classes,
        default_pairs, default_exp, telemetry_logger, exp_key,
    )
    default_summary = _build_tuned_summary(default_seed_results, task_type)
    _save_df(default_summary, _result_path(cfg_obj.output_dir, exp_key, "default_summary.csv"))

    default_best = {
        opt_name: max(
            [r for r in runs if not np.isnan(r.get(
                "test_r2" if task_type == "regression" else "test_accuracy", float("nan")
            ))],
            key=lambda r: r.get("test_r2" if task_type == "regression" else "test_accuracy", float("-inf")),
            default=runs[0] if runs else {},
        )
        for opt_name, runs in default_seed_results.items()
    }
    _save_resource_bars(default_best, f"Default [{exp_key}]", cfg_obj.output_dir, exp_key, "default_run_plots")

    # 6. Varying-seeds boxplots (reuse tuned_seed_results)
    _save_boxplots(tuned_seed_results, task_type, cfg_obj.output_dir, exp_key)

    # 7. Run history JSON
    run_history = {
        "exp_key": exp_key,
        "task_type": task_type,
        "dataset": dataset_name,
        "model_type": model_type,
        "tuned": tuned_seed_results,
        "default": default_seed_results,
    }
    _save_json(run_history, _result_path(cfg_obj.output_dir, exp_key, "run_history.json"))

    # 8. RS results CSV
    rs_rows = [
        {**r, "optimizer": opt_name}
        for opt_name, runs in tuned_seed_results.items()
        for r in runs
    ]
    if rs_rows:
        _save_df(pd.DataFrame(rs_rows), _result_path(cfg_obj.output_dir, exp_key, "rs_results.csv"))

    return {
        "tuned_seed_results": tuned_seed_results,
        "default_seed_results": default_seed_results,
        "comparison_df": comparison_df,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiments(cfg_obj: ExperimentConfig) -> dict:
    """Run the full 6-step experiment loop for all (dataset, model_type) pairs."""
    warnings.filterwarnings("ignore")

    # Initialise output directories
    _makedirs(cfg_obj.output_dir)

    # Initialise Ray + MLflow
    cfg.setup(working_dir=".", seed=cfg_obj.seeds[0], n_gpus=cfg_obj.gpu_num)
    mlflow.set_tracking_uri(cfg_obj.mlflow_uri)

    # Pick optimizer parameter grids
    task_type = cfg_obj.task_type
    optimizer_params = (
        cfg.REGRESSION_OPTIMIZERS_PARAMS
        if task_type == "regression"
        else cfg.CLASSIFICATION_OPTIMIZERS_PARAMS
    )

    # GPU resource allocation
    gpu_per_trial, cpu_per_trial, max_concurrent = cfg.detect_gpu_resources(cfg_obj.gpu_num, task_type)
    log.info("=" * 60)
    log.info("RESOURCE CONFIG  n_gpus=%d  task=%s", cfg_obj.gpu_num, task_type)
    log.info("  gpu/trial=%.2f  cpu/trial=%.1f  max_concurrent=%d",
             gpu_per_trial, cpu_per_trial, max_concurrent)
    log.info("=" * 60)

    all_results: dict[str, dict] = {}

    for dataset_name in cfg_obj.datasets:
        for model_type in cfg_obj.model_types:
            exp_key = f"{dataset_name}_{model_type}"
            try:
                result = _run_experiment_pair(
                    task_type, dataset_name, model_type, cfg_obj,
                    optimizer_params, gpu_per_trial, cpu_per_trial, max_concurrent,
                )
                all_results[exp_key] = result
            except Exception as e:
                log.error("Experiment %s failed: %s", exp_key, e, exc_info=True)

    # Cross-experiment summary
    from .stats import cross_experiment_summary
    if all_results:
        all_tuned = {k: v["tuned_seed_results"] for k, v in all_results.items()}
        cross_df = cross_experiment_summary(all_tuned, task_type)
        if not cross_df.empty:
            _save_df(cross_df, _result_path(cfg_obj.output_dir, "cross", "summary.csv"))

    ray.shutdown()
    log.info("\n%s\nAll %d experiments done.\n%s",
             "#" * 70, len(all_results), "#" * 70)
    return all_results
