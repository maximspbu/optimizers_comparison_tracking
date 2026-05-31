"""Central configuration: seeds, optimizer grids, device, Ray/MLflow utilities."""

from __future__ import annotations

import logging
import random
import subprocess
from pprint import pprint

import mlflow
import numpy as np
import pandas as pd
import ray
import torch
from ray import tune
from torch.optim import AdamW

from .optimizers import AdaBound

try:
    from torch_optimizer import Adahessian as AdaHessian
except ImportError:
    AdaHessian = None  # type: ignore[assignment]

try:
    from adabelief_pytorch import AdaBelief
except ImportError:
    AdaBelief = None  # type: ignore[assignment]

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
    from .optimizers import GaLoreAdamW
except ImportError:
    GaLoreAdamW = None  # type: ignore[assignment]

try:
    from Stacey.Stacey.staceyp2 import Stacey_p2
except ImportError:
    Stacey_p2 = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

SEED: int = 0
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
NUM_THREADS: int = 4
SKIP_KEYS: frozenset[str] = frozenset({"experiment_name", "tracking_uri", "seed"})

TORCH_GENERATOR: torch.Generator = torch.Generator()
TORCH_GENERATOR.manual_seed(SEED)

# Grid search over seeds (legacy runner)
SEED_GRID = tune.grid_search(list(SEEDS))

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
    if Stacey_p2 is not None:
        params[Stacey_p2] = {
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
    if Stacey_p2 is not None:
        params[Stacey_p2] = {
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

# Legacy grid-search grids (used by old runner / notebooks)
OPTIMIZERS_PARAMS: dict = {
    **(
        {AdaHessian: {
            "lr": tune.grid_search([1e-2, 1e-1, 0.15, 1.0]),
            "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
            "hessian_power": tune.grid_search([0.5, 0.75, 1.0]),
            "seed": SEED_GRID,
        }} if AdaHessian is not None else {}
    ),
    **(
        {Lion: {
            "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
            "weight_decay": tune.grid_search([1e-4, 1e-3, 1e-2]),
            "decoupled_weight_decay": tune.grid_search([False, True]),
            "use_triton": tune.grid_search([False]),
            "seed": SEED_GRID,
        }} if Lion is not None else {}
    ),
    **(
        {AdaBelief: {
            "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
            "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
            "amsgrad": tune.grid_search([False, True]),
            "weight_decouple": tune.grid_search([False, True]),
            "rectify": tune.grid_search([False, True]),
            "print_change_log": tune.grid_search([False]),
            "seed": SEED_GRID,
        }} if AdaBelief is not None else {}
    ),
    AdamW: {
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
        "amsgrad": tune.grid_search([False, True]),
        "seed": SEED_GRID,
    },
    AdaBound: {
        "lr": tune.grid_search([1e-7, 1e-6, 1e-5, 1e-4]),
        "final_lr": tune.grid_search([1e-7, 1e-6, 1e-5]),
        "gamma": tune.grid_search([1e-4, 1e-3, 1e-2]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3]),
        "amsbound": tune.grid_search([False, True]),
        "seed": SEED_GRID,
    },
}

IMAGE_CLASSIFICATION_OPTIMIZERS_PARAMS: dict = CLASSIFICATION_OPTIMIZERS_PARAMS

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


def detect_gpu_resources(n_gpus: int = 1) -> tuple[float, float, int]:
    """Return (gpu_per_trial, cpu_per_trial, max_concurrent) for Ray Tune.

    Probes NVIDIA MPS; if running falls back to 1 GPU per trial.
    With n_gpus GPUs, max_concurrent = n_gpus (one trial per GPU).
    """
    if not torch.cuda.is_available() or n_gpus == 0:
        return 0.0, 2.0, 4

    mps_enabled = False
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True, timeout=5,
        )
        probe = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="get_server_list\n",
            capture_output=True, text=True, timeout=3,
        )
        mps_enabled = probe.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    if mps_enabled:
        return 0.25, 0.5, min(n_gpus * 4, 8)
    return 1.0, 2.0, n_gpus


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def setup(
    working_dir: str = ".",
    seed: int = SEED,
    n_gpus: int = 1,
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
    ray.shutdown()
    ray.init(
        runtime_env={"env_vars": _SILENT_ENV, "working_dir": working_dir},
        logging_level=logging.ERROR,
        log_to_driver=False,
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
