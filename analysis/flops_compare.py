#! /usr/bin/env python3

from pathlib import Path
from re import T
import torch
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
import thop


@hydra.main(
    version_base="1.3.2",
    config_path="../configs",
    config_name="run/train/resnet_efficient_distill_both.yaml",
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    cfg.model.decoder.bucket_size = 4

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

    model = SLRModel(cfg, vocab).to("cpu")
    decoder = model.backbone.decoder
    test_data = next(iter(datamodule.test_dataloader()))
    tmp = torch.randn(100, 1, 512)

    time = datetime.now()
    with torch.no_grad():
        # flops, params = thop.profile(
        #     decoder, inputs=(tmp, torch.tensor([100], dtype=torch.int64)), verbose=True
        # )
        for i in range(10):
            decoder(tmp, torch.tensor([100], dtype=torch.int64))
    time = datetime.now() - time
    print(f"Time: {time}")
    # print(f"FLOPs: {flops}, Params: {params}")


if __name__ == "__main__":
    main()
