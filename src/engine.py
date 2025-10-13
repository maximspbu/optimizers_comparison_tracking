import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, Sequence, Optional, List
import torchmetrics
from torch_optimizer import Adahessian as AdaHessian
from copy import copy


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    metrics: Dict[str, torchmetrics.Metric],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float | Sequence]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model (nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A data to used for training.
        loss_fn (nn.Module): A PyTorch loss function to minimize.
        metrics (List[torchmetrics.Metric]): A list of metrics used for evaluating.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A Dict[str, float] with key-value pairs as name metric and its value.
    """
    model.train()
    metric_results: Dict[str, float | Sequence] = {"train_loss": 0.0}

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        metric_results["train_loss"] += loss.item()
        for metric in metrics:
            metrics[metric].update(y_pred, y)

        optimizer.zero_grad()

        loss.backward(create_graph=True) if isinstance(
            optimizer, AdaHessian
        ) else loss.backward()

        optimizer.step()

    metric_results["train_loss"] /= len(dataloader)

    for metric in metrics:
        metric_computed = metrics[metric].compute()
        metric_results["train_" + metric] = (
            metric_computed.item()
            if not isinstance(metric_computed, tuple) and metric_computed.numel() == 1
            else metric_computed
        )

        metrics[metric].reset()

    return metric_results


def valid_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    metrics: Dict[str, torchmetrics.Metric],
    device: torch.device,
) -> Dict[str, float | Sequence]:
    """Validates a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a validating dataset.

    Args:
        model (nn.Module): A PyTorch model to be tested.
        dataloader (torch.utils.data.DataLoader): A data to used for validating.
        loss_fn (nn.Module): A PyTorch loss function to minimize.
        metrics (List[torchmetrics.Metric]): A list of metrics used for evaluating.
        device (torch.device): A target device to compute on ("cuda", "cpu", "mps").

    Returns:
        A Dict[str, float] with key-value pairs as name metric and its value.
    """
    model.eval()
    metric_results: Dict[str, float | Sequence] = {"valid_loss": 0.0}

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            loss = loss_fn(test_pred, y)

            metric_results["valid_loss"] += loss.item()
            for metric in metrics:
                metrics[metric].update(test_pred, y)

    metric_results["valid_loss"] /= len(dataloader)
    for metric in metrics:
        metric_computed = metrics[metric].compute()
        metric_results["valid_" + metric] = (
            metric_computed.item()
            if not isinstance(metric_computed, tuple) and metric_computed.numel() == 1
            else metric_computed
        )
        metrics[metric].reset()
    return metric_results


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    metrics: Dict[str, torchmetrics.Metric],
    epochs: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    writer: Optional[SummaryWriter] = None,
    epoch_step: int = 10,
) -> Dict[str, List]:
    """Trains and valids a PyTorch model.

    Passes a target PyTorch models through train_step() and valid_step()
    functions for a number of epochs, training and validating the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): A data to used for training.
        valid_dataloader (torch.utils.data.DataLoader): A data to used for validating.
        test_dataloader (torch.utils.data.DataLoader): A data to used for testing.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        metrics (List[torchmetrics.Metric]): A list of metrics used for training and evaluating.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda", "cpu", "mps").
        scheduler: A LRScheduler intance for interrupting training.
        writer: A SummaryWriter intance for writing experiment results.
        epoch_step: A number to represent step count to output information about training.

    Returns:
        Dict[str, List]: A dictionary of training and validating loss as well as training and
        validating metrics.
    """
    results: Dict[str, List] = {
        "train_loss": [],
        "valid_loss": [],
    }

    for metric in metrics:
        metrics[metric] = metrics[metric].to(device)
        results[f"train_{metric}"] = []
        results[f"valid_{metric}"] = []

    for epoch in range(epochs):
        train_results = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            metrics=copy(metrics),
            optimizer=optimizer,
            device=device,
        )

        if scheduler is not None:
            assert isinstance(train_results["train_loss"], float), (
                "Train loss can be only float"
            )
            scheduler.step(train_results["train_loss"])

        valid_results = valid_step(
            model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn,
            metrics=copy(metrics),
            device=device,
        )
        if epoch % epoch_step == 0:
            for metric in train_results.keys():
                results[metric].append(train_results[metric])

            for metric in valid_results.keys():
                results[metric].append(valid_results[metric])

            if writer is not None:
                writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict=train_results | valid_results,
                    global_step=epoch,
                )

                writer.close()
            else:
                pass
    test_results = valid_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
    )

    results["test_loss"] = [test_results["valid_loss"]]
    for metric in metrics:
        results["test_" + metric] = [test_results["valid_" + metric]]
    return results
