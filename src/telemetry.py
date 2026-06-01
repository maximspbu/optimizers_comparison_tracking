"""Structured per-epoch resource telemetry written to JSONL for post-run analytics."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class TelemetryLogger:
    """Append one JSON record per epoch to a JSONL file.

    Fields per record:
        timestamp, exp_key, optimizer, seed, epoch,
        cpu_pct, ram_gb, gpu_mem_gb (list), gpu_util_pct (list)
    """

    def __init__(self, log_path: str) -> None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

    def log(
        self,
        exp_key: str,
        optimizer: str,
        seed: int,
        epoch: int,
        cpu_pct: float,
        ram_gb: float,
        gpu_mem_gb: list[float] | None = None,
        gpu_util_pct: list[float] | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exp_key": exp_key,
            "optimizer": optimizer,
            "seed": seed,
            "epoch": epoch,
            "cpu_pct": cpu_pct,
            "ram_gb": ram_gb,
            "gpu_mem_gb": gpu_mem_gb or [],
            "gpu_util_pct": gpu_util_pct or [],
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_summary(
        self,
        exp_key: str,
        optimizer: str,
        seed: int,
        system_summary: dict,
    ) -> None:
        """Log the aggregated SystemMonitorCallback summary as a single record."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exp_key": exp_key,
            "optimizer": optimizer,
            "seed": seed,
            "type": "summary",
            **system_summary,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_telemetry(log_path: str) -> pd.DataFrame:
    """Load a JSONL telemetry file into a pandas DataFrame."""
    if not os.path.exists(log_path):
        return pd.DataFrame()
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning("Skipping malformed JSONL line: %s", e)
    return pd.DataFrame(records)


def telemetry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate telemetry records: mean/std/max resource usage per optimizer.

    Supports both current summary records (avg_* scalar fields plus nested
    cpu_pct/ram_gb stats) and older per-epoch records (scalar cpu_pct/ram_gb).
    """
    if df.empty or "optimizer" not in df.columns:
        return pd.DataFrame()

    agg: dict[str, list] = {"optimizer": []}
    preferred_cols = ("avg_cpu_pct", "avg_ram_gb", "avg_gpu_gb", "avg_gpu_util_pct")
    fallback_cols = ("cpu_pct", "ram_gb")
    scalar_cols = [c for c in preferred_cols if c in df.columns]
    if not scalar_cols:
        scalar_cols = [c for c in fallback_cols if c in df.columns]

    for col in scalar_cols:
        for stat in ("mean", "std", "max"):
            agg[f"{col}_{stat}"] = []

    for opt, group in df.groupby("optimizer"):
        agg["optimizer"].append(opt)
        for col in scalar_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(vals):
                arr = vals.to_numpy(dtype=float)
                agg[f"{col}_mean"].append(float(np.mean(arr)))
                agg[f"{col}_std"].append(float(np.std(arr)))
                agg[f"{col}_max"].append(float(np.max(arr)))
            else:
                agg[f"{col}_mean"].append(float("nan"))
                agg[f"{col}_std"].append(float("nan"))
                agg[f"{col}_max"].append(float("nan"))

    return pd.DataFrame(agg).set_index("optimizer")
