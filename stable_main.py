import torch
import torch.nn as nn
import torchmetrics
import lightning as pl
import stable_pretraining as spt
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from model.loss import cpc_loss
from model.forward import cpc_forward
from model.backbone import cpc_backbone
from dataloader.dataset import get_esc50_dataloaders

def create_datamodule(data_cfg):
    train_loader, val_loader = get_esc50_dataloaders(
        meta_csv_path=data_cfg.meta_csv_path,
        audio_dir_path=data_cfg.audio_dir_path,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        seed=data_cfg.seed,
        target_sr=data_cfg.target_sr,
        window_sec=data_cfg.window_sec,
        hop_sec=data_cfg.hop_sec,
    )

    datamodule = spt.data.DataModule(train=train_loader, val=val_loader)
    return datamodule


def create_callbacks(cfg):
    linear_probe = spt.callbacks.OnlineProbe(
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(256, 50),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(50),
            "top5": torchmetrics.classification.MulticlassAccuracy(50, top_k=5),
            "f1": torchmetrics.classification.MulticlassF1Score(50),
            "precision": torchmetrics.classification.MulticlassPrecision(50),
            "recall": torchmetrics.classification.MulticlassRecall(50),
        },
    )

    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        k=10,
    )

    save_model = ModelCheckpoint(
        monitor="eval/linear_probe_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=cfg.paths.checkpoint_dir,
        filename="{epoch}-{lp_f1:.4f}",
    )

    return [linear_probe, knn_probe, save_model]

def create_module(cfg):
    backbone = cpc_backbone()
    Wk = nn.ModuleList([nn.Linear(256, 512) for _ in range(cfg.model.timestep)])

    module = spt.Module(
        backbone=backbone,
        forward=cpc_forward,
        Wk=Wk,
        timestep=cfg.model.timestep,
        cpc_loss=cpc_loss(tau=cfg.model.tau),
        optim=cfg.optim,
        hparams={"model": cfg.model},
    )
    return module

def create_trainer(cfg):
    if not cfg.disable_wandb:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
        )
    else:
        wandb_logger = False
    
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        callbacks=create_callbacks(cfg),
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
    )

    return trainer


def main(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    data = create_datamodule(data_cfg=cfg.data)
    module = create_module(cfg)
    trainer = create_trainer(cfg)
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()

if __name__ == "__main__":
    main(cfg_path="configs/cpc.yaml")

