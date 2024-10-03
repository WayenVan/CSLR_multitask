from omegaconf import OmegaConf, DictConfig
import sys
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
    default="outputs/train/2024-09-28_23-35-44/config.yaml",
)
@click.option(
    "-ckpt",
    "--checkpoint",
    default="outputs/train/2024-09-28_23-35-44/epoch=65_wer-val=19.85_lr=1.00e-06_loss=42.41.ckpt",
)
@click.option("--ph14_root", default="dataset/phoenix2014-release")
@click.option("--ph14_lmdb_root", default="dataset/preprocessed/ph14_lmdb")
@click.command()
def main(config, checkpoint, ph14_root, ph14_lmdb_root):
    cfg = OmegaConf.load(config)

    dm = Ph14DataModule(
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

    torch.onnx.export(
        model.backbone.encoder,
        (torch.randn(1, 3, 20, 224, 224), torch.tensor([18], dtype=torch.int64)),
        "outputs/test.onnx",
        input_names=["video", "length"],
        output_names=["out", "t_length", "simcc_x", "simcc_y"],
        dynamic_axes={
            "video": {0: "batch", 2: "t"},
            "length": {0: "batch"},
            "out": {0: "batch", 2: "t"},
            "t_length": {0: "batch"},
            "simcc_x": {1: "batch", 0: "t"},
            "simcc_y": {1: "batch", 0: "t"},
        },
    )


if __name__ == "__main__":
    main()
    # t = torch.randn(1, 3, 20, 224, 224)
    # a, b, c, d, e = t.size()
    # print(a, b, c, d, e)
