"""PyTorch Lightning callbacks for optimizer state management and system monitoring."""

from __future__ import annotations

import logging

import numpy as np
import psutil
import pytorch_lightning as pl
import torch

log = logging.getLogger(__name__)

try:
    import pynvml  # type: ignore[import]
    pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    pynvml = None  # type: ignore[assignment]
    _PYNVML_OK = False


class ScheduleFreeOptimizerCallback(pl.Callback):
    """Handles .train()/.eval() switches required by ScheduleFree optimizers."""

    @staticmethod
    def _underlying(opt):
        return getattr(opt, "optimizer", opt)

    def on_train_epoch_start(self, trainer, pl_module):
        for opt in trainer.optimizers:
            u = self._underlying(opt)
            if hasattr(u, "train"):
                u.train()

    def on_validation_epoch_start(self, trainer, pl_module):
        for opt in trainer.optimizers:
            u = self._underlying(opt)
            if hasattr(u, "eval"):
                u.eval()

    def on_test_epoch_start(self, trainer, pl_module):
        for opt in trainer.optimizers:
            u = self._underlying(opt)
            if hasattr(u, "eval"):
                u.eval()


def _gpu_util_pct() -> list[float]:
    """Return GPU utilisation % per device via pynvml (empty list if unavailable)."""
    if not _PYNVML_OK:
        return []
    try:
        n = pynvml.nvmlDeviceGetCount()
        return [
            pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu
            for i in range(n)
        ]
    except Exception:
        return []


def _gpu_mem_gb() -> list[float]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())]


class SystemMonitorCallback(pl.Callback):
    """Collect per-epoch CPU/RAM/GPU metrics for post-run analytics.

    After training call ``summary()`` for aggregated stats, or access
    ``epochs`` for per-epoch records.
    """

    def __init__(self) -> None:
        self.epochs: list[dict] = []

    def on_train_epoch_start(self, trainer, pl_module):
        rec = {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_gb": psutil.virtual_memory().used / 1e9,
            "gpu_mem_gb": _gpu_mem_gb(),
            "gpu_util_pct": _gpu_util_pct(),
        }
        # Reset peak GPU memory after reading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.epochs.append(rec)

    def summary(self) -> dict:
        """Return min/median/max/mean/std across epochs for each resource."""
        if not self.epochs:
            return {}

        def _stat(vals: list[float]) -> dict:
            if not vals:
                return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            a = np.asarray(vals, dtype=float)
            return {
                "min": float(a.min()),
                "median": float(np.median(a)),
                "max": float(a.max()),
                "mean": float(a.mean()),
                "std": float(a.std()),
            }

        result: dict = {
            "avg_cpu_pct": float(np.mean([e["cpu_pct"] for e in self.epochs])),
            "avg_ram_gb": float(np.mean([e["ram_gb"] for e in self.epochs])),
            "avg_gpu_gb": float(np.mean([
                max(e["gpu_mem_gb"]) if e["gpu_mem_gb"] else 0.0
                for e in self.epochs
            ])),
            "avg_gpu_util_pct": float(np.mean([
                float(np.mean(e["gpu_util_pct"])) if e["gpu_util_pct"] else 0.0
                for e in self.epochs
            ])),
            "cpu_pct": _stat([e["cpu_pct"] for e in self.epochs]),
            "ram_gb": _stat([e["ram_gb"] for e in self.epochs]),
        }

        # Per-device GPU stats
        n_gpu = max((len(e["gpu_mem_gb"]) for e in self.epochs), default=0)
        for i in range(n_gpu):
            result[f"gpu{i}_mem_gb"] = _stat([
                e["gpu_mem_gb"][i] if i < len(e["gpu_mem_gb"]) else 0.0
                for e in self.epochs
            ])
            result[f"gpu{i}_util_pct"] = _stat([
                e["gpu_util_pct"][i] if i < len(e["gpu_util_pct"]) else 0.0
                for e in self.epochs
            ])

        return result
