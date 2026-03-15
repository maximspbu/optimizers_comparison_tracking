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
from torch_optimizer import Adahessian as AdaHessian


class LightningWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper supporting both classification and regression tasks.

    For classification (default): pass ``num_classes`` and omit ``metrics`` to use
    accuracy / f1score / precision / recall automatically.

    For regression: pass a ``MetricCollection`` via ``metrics``
    (e.g. ``get_regression_metrics()`` from ``src.config``).
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
        self.model = model
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.num_classes = num_classes

        self.save_hyperparameters(ignore=["model", "loss_fn", "optimizer_class", "metrics"])

        self.automatic_optimization = False

        if metrics is None:
            metrics = MetricCollection(
                {
                    "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
                    "f1score": MulticlassF1Score(num_classes=self.num_classes),
                    "precision": MulticlassPrecision(num_classes=self.num_classes),
                    "recall": MulticlassRecall(num_classes=self.num_classes),
                }
            )

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

        create_graph_needed = isinstance(opt, AdaHessian)
        self.manual_backward(loss, create_graph=create_graph_needed)

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
