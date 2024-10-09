#! /usr/bin/env python3

from lightning import LightningModule
from pathlib import Path
import torch
from torch.optim import Optimizer
from omegaconf import OmegaConf, DictConfig
import sys

sys.path.append("src")
from hydra.utils import instantiate
from csi_sign_language.utils.git import (
    save_git_diff_to_file,
    save_git_hash,
)
from csi_sign_language.models.slr_model import SLRModel
import hydra
import os

from datetime import datetime
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import callbacks
from lightning.pytorch import trainer
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import plugins
from torch.cuda.amp.grad_scaler import GradScaler


@hydra.main(
    version_base="1.3.2",
    config_path="../configs",
    config_name="run/train/resnet_transform_with_smooth.yaml",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    # set output directory
    current_time = datetime.now()
    file_name = os.path.basename(__file__)
    save_dir = os.path.join(
        "outputs", file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    cache_dir = Path(cfg.cache_dir)

    ## build module
    datamodule = instantiate(cfg.datamodule)
    vocab = datamodule.get_vocab()

    if cfg.load_weights:
        lightning_module = SLRModel.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    else:
        lightning_module = SLRModel(cfg, vocab)

    # set logger and others
    logger = build_logger()
    debug_callback = DebugCallback(logger, current_time.strftime("%Y-%m-%d_%H-%M-%S"))
    rich_callback = callbacks.RichProgressBar()
    lr_callback = callbacks.LearningRateMonitor("step", log_momentum=True)
    ckpt_callback = callbacks.ModelCheckpoint(
        save_dir,
        save_last=True,
        filename="epoch={epoch}_wer-val={val_wer:.2f}_lr={lr-Adam:.2e}_loss={train_loss:.2f}",
        monitor="val_wer",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    t = trainer.Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        # strategy='ddp',
        devices=getattr(cfg, "devices", "auto"),
        callbacks=[ckpt_callback, lr_callback, debug_callback, rich_callback],
        logger=logger,
        log_every_n_steps=50,
        max_epochs=cfg.epoch,
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        plugins=[
            plugins.MixedPrecision(
                precision="16-mixed",
                device="cuda",
                scaler=GradScaler(
                    growth_interval=100,
                ),
            ),
        ],
    )

    if t.local_rank == 0:
        if cache_dir.exists():
            raise ValueError(
                f"Cache dir {cache_dir} already exists! remove it before training"
            )
        os.makedirs(save_dir, exist_ok=True)
        # save config
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

        logger.experiment["config"].upload(os.path.join(save_dir, "config.yaml"))
        logger.experiment["checkpoint_dir"] = save_dir
        logger.experiment["sys/tags"].add(
            OmegaConf.to_container(cfg.tags, resolve=True)
        )
        # save git
        save_git_hash(os.path.join(save_dir, "git_version.bash"))
        save_git_diff_to_file(os.path.join(save_dir, "changes.patch"))

    # setup the module
    lightning_module.set_post_process(datamodule.get_post_process())
    lightning_module.set_evaluator(
        # NOTE: should be validation so it is dev!!!
        datamodule.create_evaluator(cfg.resources.ph14.root, mode="dev")
    )
    lightning_module.set_validation_cache_dir(str(cache_dir / "validate_data_cache"))
    lightning_module.set_validation_working_dir(
        str(cache_dir / f"validate_work_dir_{t.global_rank}")
    )
    t.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=cfg.checkpoint if cfg.is_resume else None,
    )
    return


def build_logger():
    logger = pl_loggers.NeptuneLogger(
        api_key=os.getenv("NEPTUNE_API"),
        project="wayenvan/sign-language-multitask",
        log_model_checkpoints=False,
        capture_stdout=False,
    )
    return logger


class DebugCallback(Callback):
    # NOTE: this is a debug callback so no need to be efficient
    def __init__(self, logger, logdir) -> None:
        super().__init__()
        self.logger = logger
        self.tb_dir = os.path.join("outputs/tensorboard", logdir)
        self.logger.experiment["tensorboard_dir"] = self.tb_dir

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # if trainer.local_rank == 0:
        #     self.writer = SummaryWriter(self.tb_dir)
        pass

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        # scaler: GradScaler = trainer.strategy.precision_plugin.scaler
        # scale = scaler.get_scale()
        # self.logger.experiment['training/scaler'].append(scale)

        a = 1

        # inspect gradient here
        # if trainer.local_rank == 0:
        #     if trainer.global_step % 500 == 0:
        #         for name, p in pl_module.named_parameters():
        #             if 'backbone' in name:
        #                 self.writer.add_histogram(name, p.clone().detach().cpu().numpy(), trainer.global_step)
        #                 if p.grad is None:
        #                     print(f'none name: {name}', file=sys.stderr)
        #                     return
        #                 if not torch.isnan(p.grad).any():
        #                     self.writer.add_histogram(name+'/grad', p.grad.clone().detach().cpu().numpy(), trainer.global_step)
        #                 else:
        #                     print(f'nan occured, name: {name}', file=sys.stderr)
        pass

    def on_train_epoch_end(self, trainer: Trainer, pl_module: SLRModel) -> None:
        pass
        # if trainer.current_epoch < 5:
        #     ids = pl_module.train_ids_epoch
        #     rank = trainer.local_rank
        #     print(rank)
        #     self.logger.experiment[f'training/train_ids_rank{rank}'].append(f'----epoch{trainer.current_epoch}')
        #     for id in ids:
        #         self.logger.experiment[f'training/train_ids_rank{rank}'].append(f'{id}')

    def on_validation_end(self, trainer: Trainer, pl_module: SLRModel) -> None:
        pass
        # ids = pl_module.val_ids_epoch
        # rank = trainer.local_rank
        # print(rank)
        # self.logger.experiment[f'training/val_ids_rank{rank}'].append(f'----epoch{trainer.current_epoch}')
        # for id in ids:
        #     self.logger.experiment[f'training/val_ids_rank{rank}'].append(f'{id}\n')


if __name__ == "__main__":
    main()
