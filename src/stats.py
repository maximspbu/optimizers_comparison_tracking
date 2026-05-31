"""Aggregate per-seed Ray Tune / MLflow results into summary tables."""

from __future__ import annotations

import statistics
from typing import Iterable

import numpy as np
import pandas as pd


def _non_seed_hparam_columns(
    df: pd.DataFrame, hparam_columns: Iterable[str]
) -> list[str]:
    return [c for c in hparam_columns if c != "seed" and c in df.columns]


def aggregate_seeds(
    df: pd.DataFrame,
    metric: str,
    hparam_columns: Iterable[str],
    grouping_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Collapse the seed axis, returning min/median/max/std per config.

    Args:
        df: one row per trial; must contain ``metric`` and every column in
            ``hparam_columns`` and ``grouping_columns``.
        metric: metric column to aggregate (e.g. ``"val_accuracy"``).
        hparam_columns: optimizer hyperparameter columns to group on, seed excluded.
        grouping_columns: extra scenario columns (dataset, model, train_fraction, optimizer).

    Returns:
        DataFrame indexed by union of grouping_columns and non-seed hparams,
        with columns min, median, max, std, n.
    """
    if metric not in df.columns:
        raise KeyError(f"metric {metric!r} not in dataframe columns: {list(df.columns)}")

    group_cols = list(grouping_columns) + _non_seed_hparam_columns(df, hparam_columns)
    if not group_cols:
        raise ValueError("at least one grouping or non-seed hparam column is required")

    grouped = df.groupby(group_cols, dropna=False)[metric]
    return grouped.agg(
        min="min", median="median", max="max", std="std", n="count"
    ).reset_index()


def best_config_per_scenario(
    aggregated: pd.DataFrame,
    scenario_columns: Iterable[str],
    stat: str = "median",
    mode: str = "min",
) -> pd.DataFrame:
    """Pick the best hparam config per scenario by ``stat`` under ``mode``."""
    if stat not in aggregated.columns:
        raise KeyError(f"stat {stat!r} not in aggregated columns")
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    ascending = mode == "min"
    return (
        aggregated.sort_values(stat, ascending=ascending)
        .groupby(list(scenario_columns), dropna=False, as_index=False)
        .head(1)
    )


def cross_experiment_summary(
    all_tuned_results: dict[str, dict[str, list[dict]]],
    task_type: str,
) -> pd.DataFrame:
    """Build a cross-experiment pivot table from tuned multi-seed results.

    Args:
        all_tuned_results: {exp_key: {optimizer_name: [seed_result, ...]}}
        task_type: "regression" | "tabular_classification" | "image_classification"

    Returns:
        DataFrame with optimizers as index, exp_keys as columns, median primary
        metric as values.
    """
    if task_type == "regression":
        primary = "test_r2"
        secondary = "test_rmse"
    else:
        primary = "test_accuracy"
        secondary = None

    rows = []
    for exp_key, seed_results in all_tuned_results.items():
        # Infer dataset/model from exp_key  (dataset_model_type)
        for opt_name, runs in seed_results.items():
            vals = [r.get(primary, float("nan")) for r in runs
                    if not np.isnan(r.get(primary, float("nan")))]
            row: dict = {
                "experiment": exp_key,
                "optimizer":  opt_name,
                f"{primary}_median": statistics.median(vals) if vals else float("nan"),
                f"{primary}_max":    max(vals)               if vals else float("nan"),
                "n_seeds": len(runs),
            }
            if secondary:
                sec_vals = [r.get(secondary, float("nan")) for r in runs
                            if not np.isnan(r.get(secondary, float("nan")))]
                row[f"{secondary}_median"] = statistics.median(sec_vals) if sec_vals else float("nan")
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    try:
        pivot = df.pivot_table(
            index="optimizer", columns="experiment", values=f"{primary}_median"
        )
        return pivot
    except Exception:
        return df
