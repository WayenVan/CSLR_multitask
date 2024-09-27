import os
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
import click
from typing import Tuple

from tqdm import tqdm
from typing import List, Union
import cv2
from lmdb import Environment
import json

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.utils.lmdb_tool import store_data
else:
    from ...utils.lmdb_tool import store_data


@click.command()
@click.option("--data_root", default="./dataset/PHOENIX-2014-T-release-v3")
@click.option("--output_dir", default="./dataset/preprocessed/ph14t_lmdb")
@click.option("--n_threads", default=5, help="number of process to create")
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
        name="ph14t_lmdb",
        email="wayenvan@outlook.com",
        vocab=[],
    )
    _data_root = Path(data_root)
    _annotation_dir = _data_root / "PHOENIX-2014-T" / "annotations" / "manual"
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    info_file_path = _output_dir / "info.json"
    with Pool(n_threads) as p:
        for mode in ["test", "dev", "train"]:
            lmdb_database_name = _output_dir / f"{mode}"
            lmdb_env = Environment(str(lmdb_database_name), map_size=int(1e12))

            annotation_file = _annotation_dir / f"PHOENIX-2014-T.{mode}.corpus.csv"
            annotation = pd.read_csv(annotation_file, sep="|")

            feature_root = (
                _data_root / "PHOENIX-2014-T/features/fullFrame-210x260px" / mode
            )

            ids = annotation["name"].to_list()
            frame_ids = []
            frame_paths = []
            for id in ids:
                frame_id, frames_path = resolve_video_by_id(feature_root, id)
                frame_ids.extend(frame_id)
                frame_paths.extend(frames_path)

            # threading pool handle aysnchronous processing
            results = p.imap_unordered(
                lambda arg: save_single_frame(
                    arg[0], arg[1], lmdb_env=lmdb_env, image_size=image_size
                ),
                zip(frame_paths, frame_ids),
            )
            for _ in tqdm(results, total=len(frame_ids)):
                pass
            lmdb_env.close()
        print(f"Finish processing {mode} data")

    print("generating vocab")
    info["vocab"] = generate_vocab(_data_root, specials)
    with info_file_path.open("w") as f:
        json.dump(info, f)


def generate_vocab(data_root: Path, specials: Union[None, List[str]] = None):
    annotation_dir = data_root / "PHOENIX-2014-T/annotations/manual"
    glosses = set()
    for mode in ["test", "dev", "train"]:
        annotation_file = annotation_dir / f"PHOENIX-2014-T.{mode}.corpus.csv"
        annotation = pd.read_csv(annotation_file, sep="|")

        gloss: str
        for gloss in annotation["orth"]:
            glosses.update(gloss.split())

    glosses = list(glosses)
    if specials is not None:
        glosses = specials + glosses
    return glosses


def save_single_frame(
    frames_path: Path, frame_id: str, lmdb_env: Environment, image_size: Tuple[int, int]
):
    frame_data = cv2.cvtColor(cv2.imread(str(frames_path)), cv2.COLOR_BGR2RGB)
    frame_data = cv2.resize(frame_data, image_size)
    store_data(lmdb_env, frame_id, frame_data)
    return True


def resolve_video_by_id(feature_root: Path, id: str):
    frames_path = sorted(
        Path(feature_root).glob(f"{id}/*.png"),
        key=lambda x: int(
            os.path.splitext(os.path.basename(x))[0].replace("images", "")
        ),
    )
    frame_id = [id + "/" + frame.name for frame in frames_path]
    return frame_id, frames_path


if __name__ == "__main__":
    frame_id, frames_path = resolve_video_by_id(
        Path(
            "dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
        ),
        "01April_2010_Thursday_heute-6694",
    )
    print(len(frame_id), len(frames_path))
