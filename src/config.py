import ray
from ray import tune
import numpy as np
import torch
import pandas as pd
from pprint import pprint
import random

from torch.optim import AdamW
from .optimizers import AdaBound
from torch_optimizer import Adahessian as AdaHessian
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion


SEED: int = 0
NUM_THREADS: int = 4
NUM_GPUS: int = 2
TORCH_GENERATOR: torch.Generator = torch.Generator()
TORCH_GENERATOR.manual_seed(SEED)
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
