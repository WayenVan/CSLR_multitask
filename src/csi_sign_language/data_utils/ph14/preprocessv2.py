"""
now only multi-signer is supported
"""

import os
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
import click
from typing import Tuple
from functools import partial

from tqdm import tqdm
from typing import List, Union
import cv2
from lmdb import Environment
import json
from torchtext.vocab import build_vocab_from_iterator

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.utils.lmdb_tool import store_data
else:
    from ...utils.lmdb_tool import store_data


@click.command()
@click.option("--data_root", default="./dataset/phoenix2014-release")
@click.option("--output_dir", default="./dataset/preprocessed/ph14_lmdb")
@click.option("--n_threads", default=12, help="number of process to create")
@click.option("--specials", default=["<blank>"], multiple=True)
@click.option(
    "--image_size",
    type=(int, int),
    default=(210, 260),
)
def main(
    data_root: str,
    output_dir: str,
    n_threads: int,
    specials: List[str],
    image_size: Tuple[int, int],
):
    info = dict(
        name="ph14_lmdb",
        email="wayenvan@outlook.com",
        vocab=[],
    )
    _data_root = Path(data_root)
    _annotation_dir = _data_root / "phoenix-2014-multisigner/annotations/manual"
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    info_file_path = _output_dir / "info.json"
    with Pool(n_threads) as p:
        for mode in ["test", "dev", "train"]:
            lmdb_database_name = _output_dir / f"{mode}"
            annotation_file = _annotation_dir / f"{mode}.corpus.csv"
            annotation = pd.read_csv(annotation_file, sep="|")

            feature_root = (
                _data_root
                / "phoenix-2014-multisigner/features/fullFrame-210x260px"
                / mode
            )

            ids = annotation["id"].to_list()
            frame_ids = []
            frame_paths = []
            for id in ids:
                frame_id, frames_path = resolve_video_by_id(feature_root, id)
                frame_ids.extend(frame_id)
                frame_paths.extend(frames_path)

            results = p.imap_unordered(
                partial(
                    task, lmdb_env_path=str(lmdb_database_name), image_size=image_size
                ),
                list(zip(frame_paths, frame_ids)),
            )
            for _ in tqdm(results, total=len(frame_ids)):
                pass
        print(f"Finish processing {mode} data")

    print("generating vocab")
    vocab, _ = generate_vocab(str(_data_root), specials)
    info["vocab"] = vocab.get_itos()
    with info_file_path.open("w") as f:
        json.dump(info, f)


def task(arg, lmdb_env_path, image_size):
    lmdb_env = Environment(lmdb_env_path, map_size=int(1e12))
    result = save_single_frame(arg[0], arg[1], lmdb_env=lmdb_env, image_size=image_size)
    lmdb_env.close()
    return result


def save_single_frame(
    frames_path: Path, frame_id: str, lmdb_env: Environment, image_size: Tuple[int, int]
):
    frame_data = cv2.cvtColor(cv2.imread(str(frames_path)), cv2.COLOR_BGR2RGB)
    frame_data = cv2.resize(frame_data, image_size)
    store_data(lmdb_env, frame_id, frame_data)
    return True


def resolve_video_by_id(feature_root: Path, id: str):
    p = list(Path(feature_root).glob(f"{id}/1/*.png"))
    frames_path = sorted(
        p,
        key=lambda x: int(x.name[-12:-6]),
    )
    frame_id = [id + "/" + frame.name for frame in frames_path]
    return frame_id, frames_path


def create_glossdictionary(annotations, specials):
    def tokens():
        for annotation in annotations["annotation"]:
            yield annotation.split()

    vocab = build_vocab_from_iterator(tokens(), special_first=True, specials=specials)
    return vocab


def generate_vocab(data_root, specials):
    print(os.getcwd())
    annotation_file_multi = os.path.join(
        data_root, "phoenix-2014-multisigner/annotations/manual"
    )
    annotation_file_single = os.path.join(
        data_root, "phoenix-2014-signerindependent-SI5/annotations/manual"
    )

    multi = []
    si5 = []
    for type in ("dev", "train", "test"):
        multi.append(
            pd.read_csv(
                os.path.join(annotation_file_multi, type + ".corpus.csv"), delimiter="|"
            )
        )
        si5.append(
            pd.read_csv(
                os.path.join(annotation_file_single, type + ".SI5.corpus.csv"),
                delimiter="|",
            )
        )

    vocab_multi = create_glossdictionary(pd.concat(multi), specials)
    vocab_single = create_glossdictionary(pd.concat(si5), specials)

    return vocab_multi, vocab_single


if __name__ == "__main__":
    main()
