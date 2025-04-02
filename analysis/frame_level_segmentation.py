#! /usr/bin/env python3
from omegaconf import OmegaConf, DictConfig
import sys
from pathlib import Path
import pickle
import torch
import pickle

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
    default="outputs/train/2024-09-25_04-13-06/config.yaml",
    # default="outputs/train/2024-10-09_04-21-36/config.yaml",
)
@click.option(
    "-ckpt",
    "--checkpoint",
    default="outputs/train/2024-09-25_04-13-06/epoch=125_wer-val=19.43_lr=1.00e-09_loss=7.76.ckpt",
    # default="outputs/train/2024-10-09_04-21-36/epoch=48_wer-val=26.60_lr=1.00e-05_loss=0.00.ckpt",
)
@click.option("--ph14_root", default="dataset/phoenix2014-release")
@click.option("--ph14_lmdb_root", default="dataset/preprocessed/ph14_lmdb")
@click.option("--working_dir", default="outputs/evaluate_working_dir")
@click.option("--mode", default="val")
@click.command()
def main(config, checkpoint, ph14_root, ph14_lmdb_root, working_dir, mode):
    device = "cuda:0"
    if mode not in ["val", "test"]:
        raise NotImplementedError()

    cfg = OmegaConf.load(config)

    with open("outputs/test.pkl", "wb") as f:
        pickle.dump(cfg, f)

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
    ).to(device)

    dataloaders = dm.test_dataloader()
    ids = []
    result = []
    with torch.no_grad():
        for data in dataloaders:
            video = data["video"].to(device)
            print(video.shape)
            t_length = data["video_length"].to(device)
            model_out = model.backbone(video, t_length)
            out = model_out.out
            t_length = model_out.t_length

            B = video.shape[0]
            for i in range(B):
                result.append(out[: t_length[i], i, :].cpu().numpy())
                ids.append(data["id"][i])

        import numpy as np

        with open("outputs/res.pkl", "wb") as f:
            pickle.dump(
                {
                    "ids": ids,
                    "result": result,
                },
                f,
            )


if __name__ == "__main__":
    main()
