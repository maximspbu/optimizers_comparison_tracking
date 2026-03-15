import pytorch_lightning as pl


class ScheduleFreeOptimizerCallback(pl.Callback):
    """Handles the .train() / .eval() switches required by ScheduleFree optimizers
    (e.g. AdamWScheduleFree).
    """

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.optimizers:
            for opt in trainer.optimizers:
                if hasattr(opt, "train"):
                    opt.train()

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.optimizers:
            for opt in trainer.optimizers:
                if hasattr(opt, "eval"):
                    opt.eval()

    def on_test_epoch_start(self, trainer, pl_module):
        if trainer.optimizers:
            for opt in trainer.optimizers:
                if hasattr(opt, "eval"):
                    opt.eval()
