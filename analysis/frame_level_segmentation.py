#! /usr/bin/env python3
from omegaconf import OmegaConf, DictConfig
import sys
from pathlib import Path
import pickle
import torch

sys.path.append("src")
from hydra.utils import instantiate
from csi_sign_language.data.datamodule.ph14 import Ph14DataModule
from csi_sign_language.models.slr_model import SLRModel
import click
from lightning.pytorch.trainer import Trainer


# output the result of test dataset as much as possible


@click.option(
    "--config",
    "-c",
    default="outputs/train/2024-10-10_01-22-13/config.yaml",
)
@click.option(
    "-ckpt",
    "--checkpoint",
    default="outputs/train/2024-10-10_01-22-13/epoch=46_wer-val=22.19_lr=1.00e-05_loss=0.00.ckpt",
)
@click.option("--ph14_root", default="dataset/phoenix2014-release")
@click.option("--ph14_lmdb_root", default="dataset/preprocessed/ph14_lmdb")
@click.option("--working_dir", default="outputs/evaluate_working_dir")
@click.option("--mode", default="val")
@click.command()
def main(config, checkpoint, ph14_root, ph14_lmdb_root, working_dir, mode):
    if mode not in ["val", "test"]:
        raise NotImplementedError()
    cfg = OmegaConf.load(config)

    dm = Ph14DataModule(
        ph14_root,
        ph14_lmdb_root,
        batch_size=2,
        num_workers=6,
        train_shuffle=True,
        val_transform=instantiate(cfg.transforms.test),
        test_transform=instantiate(cfg.transforms.test),
    )
    model = SLRModel.load_from_checkpoint(
        checkpoint, cfg=cfg, map_location="cpu", ctc_search_type="beam", strict=False
    )

    # set options
    model.set_post_process(dm.get_post_process())
    model.set_test_working_dir(working_dir)

    t = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0],
        logger=False,
        enable_checkpointing=False,
        precision=32,
    )

    dataloaders = dm.test_dataloader()
    test_data = next(iter(dataloaders))
    print(test_data)
    video = test_data["video"]
    t_length = test_data["video_length"]
    out = model.backbone(video, t_length)


if __name__ == "__main__":
    main()
