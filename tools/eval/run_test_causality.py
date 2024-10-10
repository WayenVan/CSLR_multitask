#! /usr/bin/env python3
from omegaconf import OmegaConf, DictConfig
import sys
from pathlib import Path
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
    default="outputs/train/2024-09-28_23-35-44/config.yaml",
)
@click.option(
    "-ckpt",
    "--checkpoint",
    default="outputs/train/2024-09-28_23-35-44/epoch=65_wer-val=19.85_lr=1.00e-06_loss=42.41.ckpt",
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
        batch_size=1,
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

    # clearn folder before rning
    clean_folder(working_dir)

    t = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0],
        logger=False,
        enable_checkpointing=False,
        precision=32,
    )

    loader = dm.test_dataloader() if mode == "test" else dm.val_dataloader()

    for batch in loader:
        id, video, gloss, video_length, gloss_length, gloss_gt = model._extract_batch(
            batch
        )
        assert video.size(0) == 1, "batch size should be 1"
        video = video.squeeze(0)
        video_feeder = create_feeder(video)

        cached_frames = []
        frame_counter = 0
        for frame in video_feeder:
            cached_frames.append(frame)
            frame_counter += 1
            if frame_counter % 8 == 0:
                pass
                # TODO: implement this


def clean_folder(folder_path):
    folder = Path(folder_path)

    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()  # Remove file or symbolic link
            elif item.is_dir():
                for sub_item in item.rglob("*"):  # Recursively remove contents
                    if sub_item.is_file() or sub_item.is_symlink():
                        sub_item.unlink()
                    elif sub_item.is_dir():
                        sub_item.rmdir()
                item.rmdir()  # Finally remove the empty directory
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")


def create_feeder(video):
    data = video
    for i in range(len(data)):
        yield data[i]


if __name__ == "__main__":
    main()
