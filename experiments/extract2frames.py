from omegaconf import OmegaConf
import numpy as np
import torch
import sys

sys.path.append("src")
from hydra.utils import instantiate
from functools import partial
import PIL
from PIL import Image
from csi_sign_language.data.dataset.phoenix14 import MyPhoenix14DatasetV2
import click


@click.command()
@click.option("-t", type=int)
@click.option("-tp", type=int)
def main(t, tp):
    print(f"t={t}")
    print(f"tp={tp}")
    dataset = MyPhoenix14DatasetV2(
        "dataset/phoenix2014-release", "dataset/preprocessed/ph14_lmdb", "train"
    )
    video = dataset[0]["video"]
    # video = np.transpose(video, [0, 2, 3, 1])
    frame_t, frame_tp = video[t], video[tp]

    for i, frame in enumerate((frame_tp, frame_t)):
        frame = Image.fromarray(frame)
        frame.save(f"outputs/frame_{i}.jpg")


if __name__ == "__main__":
    main()
