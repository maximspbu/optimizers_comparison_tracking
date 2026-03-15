from pprint import pprint

import mlflow
import numpy as np
import pandas as pd
import random
import ray
import torch
from ray import tune
from torch.optim import AdamW

from .optimizers import AdaBound
from torch_optimizer import Adahessian as AdaHessian
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion

try:
    from AdEMAMix_Optimizer_Pytorch.AdEMAMix import AdEMAMix
except ImportError:
    AdEMAMix = None

try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None

try:
    from .optimizers import GaLoreAdamW
except ImportError:
    GaLoreAdamW = None

try:
    from Stacey.Stacey.staceyp2 import Stacey_p2
except ImportError:
    Stacey_p2 = None


SEED: int = 0
NUM_THREADS: int = 4
NUM_GPUS: int = 2
TORCH_GENERATOR: torch.Generator = torch.Generator()
TORCH_GENERATOR.manual_seed(SEED)

# Used by regression notebooks
OPTIMIZERS_PARAMS: dict = {
    AdaHessian: {
        "lr": tune.grid_search([1e-2, 1e-1, 0.15, 1.0]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
        "hessian_power": tune.grid_search([0.5, 0.75, 1.0]),
        "seed": tune.grid_search([SEED]),
    },
    Lion: {
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.grid_search([1e-4, 1e-3, 1e-2]),
        "decoupled_weight_decay": tune.grid_search([False, True]),
        "use_triton": tune.grid_search([False]),
    },
    AdaBelief: {
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
        "amsgrad": tune.grid_search([False, True]),
        "weight_decouple": tune.grid_search([False, True]),
        "rectify": tune.grid_search([False, True]),
        "print_change_log": tune.grid_search([False]),
    },
    AdamW: {
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
        "amsgrad": tune.grid_search([False, True]),
    },
    AdaBound: {
        "lr": tune.grid_search([1e-7, 1e-6, 1e-5, 1e-4]),
        "final_lr": tune.grid_search([1e-7, 1e-6, 1e-5]),
        "gamma": tune.grid_search([1e-4, 1e-3, 1e-2]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3]),
        "amsbound": tune.grid_search([False, True]),
    },
}

# Used by image-classification notebooks (CIFAR-10 / resnet18 style)
IMAGE_CLASSIFICATION_OPTIMIZERS_PARAMS: dict = {
    AdaBelief: {
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
        "amsgrad": tune.grid_search([False, True]),
        "weight_decouple": tune.grid_search([False, True]),
        "rectify": tune.grid_search([False, True]),
        "print_change_log": tune.grid_search([False]),
    },
    AdaHessian: {
        "lr": tune.grid_search([1e-2, 1e-1, 0.15, 1.0]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
        "hessian_power": tune.grid_search([0.5, 0.75, 1.0]),
        "seed": tune.grid_search([SEED]),
    },
    Lion: {
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.grid_search([1e-4, 1e-3, 1e-2]),
        "decoupled_weight_decay": tune.grid_search([False, True]),
        "use_triton": tune.grid_search([False]),
    },
    AdamW: {
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
        "amsgrad": tune.grid_search([False, True]),
    },
    AdaBound: {
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "final_lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "gamma": tune.grid_search([1e-4, 1e-3, 1e-2]),
        "weight_decay": tune.grid_search([0, 1e-4, 1e-3]),
        "amsbound": tune.grid_search([False, True]),
    },
}

# Used by tabular-classification notebooks (covtype / AdEMAMix / ScheduleFree style)
# Requires: schedulefree, galore-torch, and git-cloned AdEMAMix + Stacey repos
CLASSIFICATION_OPTIMIZERS_PARAMS: dict = {
    Lion: {
        "lr": tune.grid_search([1e-3]),
        "weight_decay": tune.grid_search([0, 1e-5]),
        "decoupled_weight_decay": tune.grid_search([False, True]),
        "use_triton": tune.grid_search([False]),
    },
    **(
        {
            AdEMAMix: {
                "lr": tune.grid_search([1e-2]),
                "weight_decay": tune.grid_search([0, 1e-5]),
                "alpha": tune.grid_search([10.0, 15.0]),
            }
        }
        if AdEMAMix is not None
        else {}
    ),
    **(
        {
            AdamWScheduleFree: {
                "lr": tune.grid_search([1e-1, 0.25]),
                "weight_decay": tune.grid_search([1e-2, 1e-1]),
                "r": tune.grid_search([1.0]),
                "weight_lr_power": tune.grid_search([0.5, 1.0]),
            }
        }
        if AdamWScheduleFree is not None
        else {}
    ),
    **(
        {
            GaLoreAdamW: {
                "lr": tune.grid_search([1e-2]),
                "weight_decay": tune.grid_search([1e-5, 1e-4]),
                "correct_bias": tune.grid_search([False, True]),
                "no_deprecation_warning": tune.grid_search([True]),
            }
        }
        if GaLoreAdamW is not None
        else {}
    ),
    **(
        {
            Stacey_p2: {
                "lr_tau": tune.grid_search([1e-3, 1e-2, 1e-1]),
                "lr_eta": tune.grid_search([1e-3, 1e-2, 1e-1]),
                "lr_alpha": tune.grid_search([1e-3, 1e-2, 1e-1]),
                "weight_decay": tune.grid_search([0, 1e-4, 1e-3, 1e-2]),
            }
        }
        if Stacey_p2 is not None
        else {}
    ),
    AdamW: {
        "lr": tune.grid_search([1e-2]),
        "weight_decay": tune.grid_search([1e-3]),
        "amsgrad": tune.grid_search([False]),
    },
}

DEVICE: torch.device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def setup(working_dir: str = ".") -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pd.set_option("display.max_colwidth", None)
    ray.shutdown()
    ray.init(
        runtime_env={
            "env_vars": {"RAY_AIR_RICH_LAYOUT": "1"},
            "working_dir": working_dir,
        },
    )
    resources = ray.cluster_resources()
    pprint(resources)
    torch.set_num_threads(NUM_THREADS)


def seed_worker(worker_id: int):
    np.random.seed(SEED)
    random.seed(SEED)


def get_or_create_experiment(experiment_name: str) -> str:
    """Retrieve the ID of an existing MLflow experiment or create a new one.

    Args:
        experiment_name: Name of the MLflow experiment.

    Returns:
        ID of the existing or newly created MLflow experiment.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def get_regression_metrics():
    """Return a MetricCollection for regression tasks (MAPE + R2Score).

    Pass the result as the ``metrics`` argument to ``LightningWrapper``
    when training regression models.
    """
    from torchmetrics import MetricCollection
    from torchmetrics.regression import MeanAbsolutePercentageError, R2Score

    return MetricCollection({"mape": MeanAbsolutePercentageError(), "r2score": R2Score()})
