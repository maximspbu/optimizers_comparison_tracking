from datetime import datetime
from typing import Optional
import os
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import sys
from torchmetrics.classification import MulticlassROC
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter


def create_writer(
    experiment_name: str,
    model_name: str,
    extra: Optional[str] = None,
) -> SummaryWriter:
    """Creates a SummaryWriter instance saving to log_dir.
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.
    timestamp at format YYYY-MM-DD.
    Args:
        experiment_name (str): Name of the experiment.
        model_name (str): Name of model.
        extra (Optional[str]): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter: SummaryWriter for tracking experiment results.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def plot_results(
    results: dict,
    exclude_metrics: list | None = None,
    epoch_step: int = 5,
    test_num_epochs: int = 301,
    epoch_limit: int = 50,
):
    if exclude_metrics is None:
        exclude_metrics = []
    plot_data = pd.DataFrame(results)
    plot_data = plot_data.drop(
        plot_data.index[plot_data.index.to_series().str.startswith("test_")], axis=0
    )
    optimizer_names = list(results.keys())
    plot_data.columns = optimizer_names

    f, axs = plt.subplots(
        len(plot_data.index) - len(exclude_metrics), 1, figsize=(16, 32), squeeze=False
    )
    x_epochs = list(range(0, test_num_epochs, epoch_step))

    for s, metric in enumerate(plot_data.index):
        if metric in exclude_metrics:
            continue
        metric_runs_raw = plot_data.loc[metric].values
        y_values_np = (
            np.stack(metric_runs_raw)
            if not isinstance(metric_runs_raw, torch.Tensor)
            else np.stack([i for i in metric_runs_raw.cpu()])
        )
        df_to_plot = pd.DataFrame(
            y_values_np.T, index=x_epochs, columns=optimizer_names
        )
        sns.lineplot(
            data=df_to_plot,
            ax=axs[s, 0],
        )

        axs[s, 0].set_title(f"{metric} plot")
        axs[s, 0].set_xlabel("Epoch")
        # axs[s, 0].set_xlim(left=0, right=epoch_limit)
        axs[s, 0].set_ylabel(metric)
        axs[s, 0].grid(True)
        axs[s, 0].legend(title="Optimizer")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_results(results: dict):
    f, axs = plt.subplots(len(results), 1, figsize=(8, 16), squeeze=False)

    for s, optimizer_type in enumerate(results):
        train_cm = results[optimizer_type]["test_confusion_matrix"][0].cpu()
        sns.heatmap(train_cm, annot=True, ax=axs[s, 0], fmt=".20g")
        axs[s, 0].set_title(f"{optimizer_type} test confusion matrix")

    plt.tight_layout()
    plt.show()


def test_plot_roc_results(results: dict, num_classes: int):
    fig_height = 6 * len(results)
    f, axs = plt.subplots(len(results), 1, figsize=(12, fig_height), squeeze=False)

    for s, optimizer_type in enumerate(results):
        current_ax = axs[s, 0]
        metric = MulticlassROC(num_classes=num_classes)
        curve_input_data = results[optimizer_type]["test_multiclass_roc"][0]

        try:
            preds, target = curve_input_data
            metric.update(preds, target)
            computed_data, returned_ax = metric.plot(score=True, ax=current_ax)

        except ValueError:
            print(
                f"Warning: Assuming curve data for {optimizer_type} is (fpr, tpr, thresholds). Manually calculating AUC."
            )
            from sklearn.metrics import auc

            fpr_list, tpr_list, _ = curve_input_data
            computed_data = {}

            returned_ax = metric.plot(
                curve=curve_input_data, score=False, ax=current_ax
            )[1]

            auc_scores = []
            if isinstance(fpr_list, list) and isinstance(tpr_list, list):
                for i in range(num_classes):
                    fpr = fpr_list[i]
                    tpr = tpr_list[i]
                    if hasattr(fpr, "cpu"):
                        fpr = fpr.cpu().numpy()
                    if hasattr(tpr, "cpu"):
                        tpr = tpr.cpu().numpy()
                    roc_auc = auc(fpr, tpr)
                    auc_scores.append(roc_auc)

                computed_data["roc_auc_per_class"] = (
                    torch.tensor(auc_scores)
                    if "torch" in sys.modules
                    else np.array(auc_scores)
                )
            else:
                print(
                    f"Could not compute AUC for {optimizer_type} due to unexpected curve data structure."
                )

        handles, original_labels = current_ax.get_legend_handles_labels()

        new_labels = []
        auc_values = None

        if isinstance(computed_data, dict):
            if "roc_auc_per_class" in computed_data:
                auc_values = computed_data["roc_auc_per_class"]
            elif "auc_per_class" in computed_data:
                auc_values = computed_data["auc_per_class"]
        elif isinstance(computed_data, (list, tuple, torch.Tensor)):
            if len(computed_data) >= num_classes:
                auc_values = computed_data

        if auc_values is not None:
            if isinstance(auc_values, torch.Tensor):
                auc_list = auc_values.tolist()
            else:
                auc_list = list(auc_values)

            if len(auc_list) >= num_classes and len(handles) >= num_classes:
                for i in range(len(handles)):
                    if i < num_classes:
                        base_label = (
                            original_labels[i]
                            if i < len(original_labels)
                            else f"Class {i}"
                        )
                        auc_score = auc_list[i]
                        new_labels.append(f"{base_label} (AUC = {auc_score:.3f})")
                    else:
                        new_labels.append(
                            original_labels[i]
                            if i < len(original_labels)
                            else f"Curve {i}"
                        )
            else:
                new_labels = original_labels
        else:
            new_labels = original_labels
        if handles:
            current_ax.legend(handles, new_labels)

        current_ax.set_title(f"{optimizer_type} test roc plot")
        current_ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
