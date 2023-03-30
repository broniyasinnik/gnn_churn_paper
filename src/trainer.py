import os
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from .settings import CONFIG_ROOT, LOGS_ROOT
from .models import MultiChannelModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from .dataset import UsersDataModule
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import torch
import numpy as np
from torchmetrics.functional.classification import confusion_matrix
from typing import Dict
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score


def result_metrics(outputs: Dict[str, torch.tensor]):
    metrics = MetricCollection([Accuracy(task='binary'),
                                Precision(task='binary'),
                                Recall(task='binary'),
                                F1Score(task='binary')
                                ])
    logits = torch.cat([batch["logits"] for batch in outputs])
    targets = torch.cat([batch["targets"] for batch in outputs])
    targets = targets.type(torch.int8)
    results = metrics(logits, targets)
    confmat = confusion_matrix(task='binary', preds=logits, target=targets, num_classes=2)
    results.update({"tn": confmat[0][0], "fp": confmat[0][1],
                    "fn": confmat[1][0], "tp": confmat[1][1]})
    results = {key: np.around(val.numpy(), 2).item() for key, val in results.items()}
    return results


def summarize_experiment_results(model: pl.LightningModule,
                                 data_module: pl.LightningDataModule,
                                 trainer: pl.Trainer,
                                 cfg: DictConfig):
    # Load the best model from checkpoint
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, cfg=cfg)
    train_results = result_metrics(trainer.predict(model, dataloaders=data_module.train_dataloader()))
    valid_results = result_metrics(trainer.predict(model, dataloaders=data_module.val_dataloader()))
    test_results = result_metrics(trainer.predict(model, dataloaders=data_module.test_dataloader()))

    # Build a table summarizing the results and save in the test folder
    test_logger = trainer.loggers[3].log_dir
    os.makedirs(test_logger, exist_ok=True)
    results = pd.DataFrame({"train": train_results,
                            "valid_fold_1": valid_results,
                            "test": test_results}).T
    results.to_csv(os.path.join(test_logger, "results.csv"), float_format='{:0.2f}'.format)


@hydra.main(version_base=None, config_path='../assets/config/', config_name='experiment_small')
def train_churn_classifier(cfg: DictConfig):
    pl.seed_everything(42)
    data = UsersDataModule(cfg=cfg)
    # TODO: Extend training and evaluation to run on multiple folds.
    data.setup(fold=1)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    model = MultiChannelModel(cfg)
    # Define loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=LOGS_ROOT / cfg.log_dir / "tb",
                                             name=cfg.experiment_name)
    csv_trn_logger = pl_loggers.CSVLogger(save_dir=LOGS_ROOT / cfg.log_dir / "train",
                                          name=cfg.experiment_name,
                                          prefix='train')
    csv_val_logger = pl_loggers.CSVLogger(save_dir=LOGS_ROOT / cfg.log_dir / "valid",
                                          name=cfg.experiment_name,
                                          prefix='valid')

    csv_test_logger = pl_loggers.CSVLogger(save_dir=LOGS_ROOT / cfg.log_dir / "test",
                                           name=cfg.experiment_name,
                                           prefix='test')

    # saves top-1 checkpoints based on "val_loss" metric
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=tb_logger.log_dir + "/checkpoints",
        filename="gnn-{epoch:02d}-{val_loss:.2f}.bst",
    )

    # saves last epoch checkpoint
    last_checkpoint_callback = ModelCheckpoint(
        save_last=True,
        dirpath=tb_logger.log_dir + "/checkpoints",
        filename="gnn-{epoch:02d}.lst",
    )

    # Train model
    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         accelerator="auto",
                         devices=1,
                         logger=[tb_logger, csv_trn_logger, csv_val_logger, csv_test_logger],
                         callbacks=[best_checkpoint_callback,
                                    last_checkpoint_callback]
                         )

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    summarize_experiment_results(model=model,
                                 data_module=data,
                                 trainer=trainer,
                                 cfg=cfg)


if __name__ == "__main__":
    # os.environ["HYDRA_FULL_ERROR"] = "1"
    # cfg = OmegaConf.load(CONFIG_ROOT/"experiment_small.yaml")
    train_churn_classifier()
