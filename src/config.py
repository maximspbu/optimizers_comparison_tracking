"""Central configuration: seeds, optimizer grids, device, Ray/MLflow utilities."""

from __future__ import annotations

import logging
import os
import random
from pprint import pprint

import mlflow
import numpy as np
import pandas as pd
import ray
import torch
from ray import tune
from torch.optim import AdamW

try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None  # type: ignore[assignment]

try:
    from AdEMAMix_Optimizer_Pytorch.AdEMAMix import AdEMAMix
except ImportError:
    AdEMAMix = None  # type: ignore[assignment]

try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None  # type: ignore[assignment]

try:
    from .optimizers import GaLoreAdamW, Stacey_pp
except ImportError:
    GaLoreAdamW = None  # type: ignore[assignment]
    Stacey_pp = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

SEED: int = 0
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
NUM_THREADS: int = 4
SKIP_KEYS: frozenset[str] = frozenset({"experiment_name", "tracking_uri", "seed"})
RAY_RUNTIME_ENV_EXCLUDES: tuple[str, ...] = (
    "/.DS_Store",
    "/.ipynb_checkpoints/",
    "/.jupyter/",
    "/.jupyter_ystore.db",
    "/.ruff_cache/",
    "/.venv/",
    "/analyzes/",
    "/data/",
    "/mlruns/",
    "/outputs/",
    "/review/",
    "*.bak",
    "*.tmp",
)
REQUIRED_OPTIMIZER_NAMES: tuple[str, ...] = (
    "AdamW",
    "Stacey_pp",
    "Lion",
    "AdamWScheduleFree",
    "GaLoreAdamW",
    "AdEMAMix",
)

TORCH_GENERATOR: torch.Generator = torch.Generator()
TORCH_GENERATOR.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Optimizer grids – Regression (notebook-style, Optuna/loguniform)
# ---------------------------------------------------------------------------

def _build_regression_params() -> dict:
    params: dict = {}
    if Lion is not None:
        params[Lion] = {
            "lr": tune.loguniform(5e-5, 5e-2),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3, 1e-2]),
            "decoupled_weight_decay": tune.choice([False, True]),
        }
    if AdEMAMix is not None:
        params[AdEMAMix] = {
            "lr": tune.loguniform(1e-4, 2e-1),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
            "alpha": tune.uniform(2.0, 20.0),
        }
    if AdamWScheduleFree is not None:
        params[AdamWScheduleFree] = {
            "lr": tune.loguniform(5e-4, 5e-1),
            "weight_decay": tune.choice([0, 1e-4, 1e-3, 1e-2, 1e-1]),
            "r": tune.uniform(0.0, 1.0),
            "weight_lr_power": tune.uniform(0.5, 2.0),
        }
    if GaLoreAdamW is not None:
        params[GaLoreAdamW] = {
            "lr": tune.loguniform(1e-4, 5e-1),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
            "correct_bias": tune.choice([False, True]),
        }
    if Stacey_pp is not None:
        params[Stacey_pp] = {
            "lr_tau": tune.loguniform(1e-4, 1.0),
            "lr_eta": tune.loguniform(1e-4, 1.0),
            "lr_alpha": tune.loguniform(1e-4, 1.0),
            "weight_decay": tune.choice([0, 1e-3, 1e-2]),
        }
    params[AdamW] = {
        "lr": tune.loguniform(1e-4, 5e-1),
        "weight_decay": tune.loguniform(1e-5, 5e-2),
        "amsgrad": tune.choice([False, True]),
    }
    return params


REGRESSION_OPTIMIZERS_PARAMS: dict = _build_regression_params()

# ---------------------------------------------------------------------------
# Optimizer grids – Tabular / Image Classification (notebook-style)
# ---------------------------------------------------------------------------

def _build_classification_params() -> dict:
    params: dict = {}
    if Lion is not None:
        params[Lion] = {
            "lr": tune.loguniform(5e-5, 5e-2),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3, 1e-2]),
            "decoupled_weight_decay": tune.choice([False, True]),
        }
    if AdEMAMix is not None:
        params[AdEMAMix] = {
            "lr": tune.loguniform(1e-4, 2e-1),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
            "alpha": tune.uniform(2.0, 20.0),
        }
    if AdamWScheduleFree is not None:
        params[AdamWScheduleFree] = {
            "lr": tune.loguniform(5e-4, 5e-1),
            "weight_decay": tune.choice([0, 1e-4, 1e-3, 1e-2, 1e-1]),
            "r": tune.uniform(0.0, 1.0),
            "weight_lr_power": tune.uniform(0.5, 2.0),
        }
    if GaLoreAdamW is not None:
        params[GaLoreAdamW] = {
            "lr": tune.loguniform(1e-4, 5e-1),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
            "correct_bias": tune.choice([False, True]),
        }
    if Stacey_pp is not None:
        params[Stacey_pp] = {
            "lr_tau": tune.loguniform(1e-4, 1.0),
            "lr_eta": tune.loguniform(1e-4, 1.0),
            "lr_alpha": tune.loguniform(1e-4, 1.0),
            "weight_decay": tune.choice([0, 1e-3, 1e-2]),
        }
    params[AdamW] = {
        "lr": tune.loguniform(1e-4, 5e-1),
        "weight_decay": tune.loguniform(1e-5, 5e-2),
        "amsgrad": tune.choice([False, True]),
    }
    return params


CLASSIFICATION_OPTIMIZERS_PARAMS: dict = _build_classification_params()


def _build_image_classification_params() -> dict:
    """Conservative LR ranges for pretrained vision backbones."""
    params: dict = {}
    if Lion is not None:
        params[Lion] = {
            "lr": tune.loguniform(1e-5, 3e-4),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
            "decoupled_weight_decay": tune.choice([False, True]),
        }
    if AdEMAMix is not None:
        params[AdEMAMix] = {
            "lr": tune.loguniform(1e-5, 3e-3),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
            "alpha": tune.uniform(2.0, 12.0),
        }
    if AdamWScheduleFree is not None:
        params[AdamWScheduleFree] = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3, 1e-2]),
            "r": tune.uniform(0.0, 1.0),
            "weight_lr_power": tune.uniform(0.5, 2.0),
        }
    if GaLoreAdamW is not None:
        params[GaLoreAdamW] = {
            "lr": tune.loguniform(1e-5, 3e-3),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
            "correct_bias": tune.choice([False, True]),
        }
    if Stacey_pp is not None:
        params[Stacey_pp] = {
            "lr_tau": tune.loguniform(1e-5, 1e-2),
            "lr_eta": tune.loguniform(1e-5, 1e-2),
            "lr_alpha": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.choice([0, 1e-5, 1e-4, 1e-3]),
        }
    params[AdamW] = {
        "lr": tune.loguniform(1e-5, 3e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "amsgrad": tune.choice([False, True]),
    }
    return params


IMAGE_CLASSIFICATION_OPTIMIZERS_PARAMS: dict = _build_image_classification_params()
OPTIMIZERS_PARAMS: dict = CLASSIFICATION_OPTIMIZERS_PARAMS


def optimizer_names(optimizer_params: dict) -> list[str]:
    """Return optimizer class names in the order they will run."""
    return [optimizer_class.__name__ for optimizer_class in optimizer_params]


def validate_required_optimizers(optimizer_params: dict, task_type: str) -> None:
    """Fail fast if a required optimizer dependency was not imported."""
    present = set(optimizer_names(optimizer_params))
    missing = [name for name in REQUIRED_OPTIMIZER_NAMES if name not in present]
    extra = [name for name in present if name not in REQUIRED_OPTIMIZER_NAMES]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing required optimizers: {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected optimizers: {', '.join(sorted(extra))}")
        raise RuntimeError(
            f"Invalid optimizer set for task_type={task_type!r}: {'; '.join(parts)}. "
            "Install/import exactly: " + ", ".join(REQUIRED_OPTIMIZER_NAMES)
        )

# ---------------------------------------------------------------------------
# Device / GPU resource detection
# ---------------------------------------------------------------------------

DEVICE: torch.device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def detect_gpu_resources(
    n_gpus: int = 1,
    task_type: str = "tabular_classification",
) -> tuple[float, float, int]:
    """Return (gpu_per_trial, cpu_per_trial, max_concurrent) for Ray Tune.

    Uses fractional GPU allocation so multiple trials share each card:
      - tabular / regression  → 4 trials per GPU (MLP is tiny, ~300–500 MB)
      - image classification  → 3 trials per GPU (ResNet/EfficientNet ~2–3 GB)

    CPUs are distributed evenly; tabular tasks use num_workers=0 (data is
    in-memory), so 1–2 CPU per trial is enough.
    """
    total_cpus = os.cpu_count() or 4

    if not torch.cuda.is_available() or n_gpus == 0:
        cpu_per = max(2.0, float(total_cpus) / 4)
        return 0.0, cpu_per, 4

    trials_per_gpu = 3 if task_type == "image_classification" else 4
    max_concurrent = n_gpus * trials_per_gpu
    gpu_per_trial = 1.0 / trials_per_gpu
    cpu_per_trial = max(1.0, float(total_cpus) / max_concurrent)

    return gpu_per_trial, cpu_per_trial, max_concurrent


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def setup(
    working_dir: str = ".",
    seed: int = SEED,
    n_gpus: int = 1,
    n_cpus: int | None = None,
    silent: bool = True,
) -> None:
    """Initialise Ray, seed everything, set pandas/torch options."""
    _SILENT_ENV = {
        "RAY_AIR_RICH_LAYOUT": "1",
        "RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS": "0",
        "RAY_DEDUP_LOGS": "0",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTHONWARNINGS": "ignore",
        "ABSL_LOGGING_LOG_TO_STDERR": "0",
        "GLOG_minloglevel": "3",
    }
    if silent:
        import logging as _logging
        for _lg in ("ray", "ray.tune", "ray.air", "ray.train", "pytorch_lightning", "lightning", "absl"):
            _logging.getLogger(_lg).setLevel(_logging.ERROR)

    set_global_seed(seed)
    pd.set_option("display.max_colwidth", None)
    ray_cpus = n_cpus if n_cpus is not None else (os.cpu_count() or 4)
    ray.shutdown()
    ray.init(
        runtime_env={
            "env_vars": _SILENT_ENV,
            "working_dir": working_dir,
            "excludes": list(RAY_RUNTIME_ENV_EXCLUDES),
        },
        logging_level=logging.ERROR,
        log_to_driver=False,
        num_cpus=ray_cpus,
        num_gpus=n_gpus if n_gpus > 0 else None,
    )
    pprint(ray.cluster_resources())
    torch.set_num_threads(NUM_THREADS)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int, seed: int = SEED) -> None:
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def get_or_create_experiment(experiment_name: str) -> str:
    if exp := mlflow.get_experiment_by_name(experiment_name):
        return exp.experiment_id
    return mlflow.create_experiment(experiment_name)


def get_regression_metrics():
    """Return a MetricCollection for legacy regression (MAPE + R2Score)."""
    from torchmetrics import MetricCollection
    from torchmetrics.regression import MeanAbsolutePercentageError, R2Score

    return MetricCollection({"mape": MeanAbsolutePercentageError(), "r2score": R2Score()})
