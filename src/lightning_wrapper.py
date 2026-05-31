"""PyTorch Lightning wrappers for classification and regression tasks."""

from __future__ import annotations

import inspect

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torch_optimizer import Adahessian as AdaHessian


class LightningWrapper(pl.LightningModule):
    """Lightning wrapper for classification (and generic regression via custom metrics).

    For classification: pass ``num_classes``; accuracy/f1/precision/recall are
    built automatically.
    For regression: pass a ``MetricCollection`` via ``metrics``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class,
        optimizer_hparams: dict,
        loss_fn: nn.Module,
        metrics: MetricCollection | None = None,
        num_classes: int = 2,
    ):
        super().__init__()
        optimizer_hparams = dict(optimizer_hparams)
        seed = optimizer_hparams.pop("seed", 0)
        if "seed" in inspect.signature(optimizer_class).parameters:
            optimizer_hparams["seed"] = seed
        pl.seed_everything(seed, workers=True)

        self.model = model
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self._seed = seed
        self.save_hyperparameters({"optimizer_hparams": optimizer_hparams, "seed": seed})
        self.automatic_optimization = False

        if metrics is None:
            metrics = MetricCollection({
                "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
                "f1score": MulticlassF1Score(num_classes=self.num_classes),
                "precision": MulticlassPrecision(num_classes=self.num_classes),
                "recall": MulticlassRecall(num_classes=self.num_classes),
            })

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        opt.zero_grad()
        create_graph = isinstance(opt, AdaHessian)
        self.manual_backward(loss, create_graph=create_graph)
        opt.step()
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.train_metrics.update(logits, y)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss_step", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.update(logits, y)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(logits, y)
        return loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return self.optimizer_class(
            self.model.parameters(), **self.hparams.optimizer_hparams
        )


class RegressionLightningWrapper(pl.LightningModule):
    """Lightning wrapper for regression tasks.

    Tracks MSE, MAE, R² and additionally logs RMSE = sqrt(MSE) at epoch end.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class,
        optimizer_hparams: dict,
        loss_fn: nn.Module,
    ):
        super().__init__()
        optimizer_hparams = dict(optimizer_hparams)
        optimizer_hparams.pop("seed", None)

        self.model = model
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=["model", "loss_fn", "optimizer_class"])
        self.automatic_optimization = False

        m = MetricCollection({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(),
        })
        self.train_metrics = m.clone(prefix="train_")
        self.val_metrics = m.clone(prefix="val_")
        self.test_metrics = m.clone(prefix="test_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        opt.zero_grad()
        _inner = getattr(opt, "optimizer", opt)
        create_graph = AdaHessian is not None and isinstance(_inner, AdaHessian)
        self.manual_backward(loss, create_graph=create_graph)
        opt.step()
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.train_metrics.update(preds, y)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss_step", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.update(preds, y)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        metrics["val_rmse"] = torch.sqrt(metrics["val_mse"].clamp(min=0))
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(preds, y)
        return loss

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        metrics["test_rmse"] = torch.sqrt(metrics["test_mse"].clamp(min=0))
        self.log_dict(metrics, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return self.optimizer_class(
            self.model.parameters(), **self.hparams.optimizer_hparams
        )
