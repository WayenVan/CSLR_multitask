import os
from typing import Any
import numpy as np
import glob
import cv2 as cv2
import pandas as pd
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from torchtext.vocab import vocab, build_vocab_from_iterator, Vocab
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path

from csi_sign_language.csi_typing import PaddingMode
from ...csi_typing import *
from ...utils.data import VideoGenerator, padding, load_vocab
from typing import Literal, List, Union
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from ...utils.lmdb_tool import retrieve_data
import json
import yaml
import lmdb


class MyPhoenix14Dataset(Dataset):
    data_root: str

    def __init__(
        self,
        data_root: str,
        subset: Union[Literal["multisigner"], Literal["si5"]],
        mode: Union[Literal["train"], Literal["dev"], Literal["test"]],
        gloss_length=None,
        video_length=None,
        transform=None,
        excluded_ids=[],
    ) -> None:
        self.data_root = data_root
        self.subset = subset
        self.mode = mode
        self.subset_root = os.path.join(data_root, subset)
        with open(os.path.join(self.subset_root, "info.json"), "r") as f:
            self.info = json.load(f)
        # self.info = OmegaConf.load(os.path.join(self.subset_root, 'info.yaml'))

        self.vocab = self.create_vocab_from_list(self.info["vocab"])
        self.data_id: List[str] = [
            id for id in self.info[mode]["data"] if id not in excluded_ids
        ]

        self.transform = transform

        self.video_length = video_length
        self.gloss_length = gloss_length
        self.lmdb_env = None

    def get_stm(self):
        return os.path.join(
            self.subset_root, f"phoenix2014-groundtruth-{self.mode}.stm"
        )

    def id2index(self, id):
        return self.data_id.index(id)

    def __getitem__(self, index) -> Any:
        if self.lmdb_env is None:
            self._init_db()

        id = self.data_id[index]
        data = retrieve_data(self.lmdb_env, id)
        video = data["video"]
        video = rearrange(video, "t c h w -> t h w c")
        gloss_label = data["gloss_labels"]
        gloss = np.array(self.vocab(gloss_label), dtype="int64")

        ret = dict(
            id=id,
            video=video,  # [t c h w], uint8, 0-255
            gloss=gloss,
            gloss_label=gloss_label,
        )

        if self.transform is not None:
            ret = self.transform(ret)

        return ret

    def __len__(self):
        return len(self.data_id)

    def _init_db(self):
        self.lmdb_env = lmdb.open(
            os.path.join(self.data_root, self.subset, self.mode, "feature_database"),
            readonly=True,
            lock=False,
            create=False,
        )

    def get_vocab(self):
        return self.vocab

    @staticmethod
    def create_vocab_from_list(list: List[str]):
        return vocab(OrderedDict([(item, 1) for item in list]))


class MyPhoenix14DatasetV2(Dataset):
    """
    support mult
    """

    def __init__(
        self,
        data_root: str,
        feature_root: str,
        mode: Literal["test", "dev", "train"] = "train",
        thread_pool: Union[ThreadPoolExecutor, None] = None,
        transform=None,
        excluded_ids=[],
    ) -> None:
        self.data_root = Path(data_root)
        self.mode = mode
        self.feature_root = Path(feature_root)
        self.lmdb_root = self.feature_root / self.mode
        self.annotation_file = (
            self.data_root
            / f"phoenix-2014-multisigner/annotations/manual/{self.mode}.corpus.csv"
        )
        self.annotation = pd.read_csv(self.annotation_file, sep="|")
        # adjust data
        if len(excluded_ids) > 0:
            self.annotation = self.annotation[~self.annotation.id.isin(excluded_ids)]

        # load info file
        with open(os.path.join(self.feature_root, "info.json"), "r") as f:
            self.info = json.load(f)
        # generate vocab
        self.vocab = self.create_vocab_from_list(self.info["vocab"])
        self.transform = transform
        self.thread_pool = thread_pool

    def get_frame_ids(self, id):
        feature_root = (
            self.data_root
            / f"phoenix-2014-multisigner/features/fullFrame-210x260px/{self.mode}"
        )
        p = list(Path(feature_root).glob(f"{id}/1/*.png"))
        frames_path = sorted(
            p,
            key=lambda x: int(x.name[-12:-6]),
        )
        frame_id = [id + "/" + frame.name for frame in frames_path]
        return frame_id

    def retreive_frames(self, frame_ids):
        # NOTE: ids is sorted
        if hasattr(self, "lmdb_env") is False:
            self._init_db()

        if self.thread_pool is not None:
            # NOTE: order need to be the same
            futures = [
                self.thread_pool.submit(retrieve_data, self.lmdb_env, id)
                for id in frame_ids
            ]
            results = [future.result() for future in futures]
            return results

        results = [retrieve_data(self.lmdb_env, id) for id in frame_ids]
        return results

    def __getitem__(self, index) -> Any:
        item = self.annotation.iloc[index]
        id = item["id"]
        glosses = item["annotation"]
        glosses = glosses.split()
        glosses_index = np.array(self.vocab(glosses), dtype=np.int64)
        video = self.retreive_frames(self.get_frame_ids(id))
        video = np.stack(video, dtype=np.uint8)

        ret = dict(
            id=id,
            video=video,  # [t c h w], uint8, 0-255
            gloss=glosses_index,
            gloss_label=glosses,
        )

        if self.transform is not None:
            ret = self.transform(ret)

        return ret

    def __len__(self):
        return len(self.annotation)

    def _init_db(self):
        self.lmdb_env = lmdb.open(
            str(self.lmdb_root),
            readonly=True,
            lock=False,
            create=False,
        )

    def get_vocab(self):
        return self.vocab

    @staticmethod
    def create_vocab_from_list(list: List[str]):
        return vocab(OrderedDict([(item, 1) for item in list]))


class CollateFn:
    def __init__(
        self, gloss_mode="padding", length_video=None, length_gloss=None
    ) -> None:
        self.length_video = length_video
        self.length_gloss = length_gloss
        self.gloss_mode = gloss_mode

    def __call__(self, data) -> Any:
        # sort the data by video length in decreasing way for onxx
        data = sorted(data, key=lambda x: len(x["video"]), reverse=True)

        video_batch = [item["video"] for item in data]
        gloss_batch = [item["gloss"] for item in data]
        gloss_label = [item["gloss_label"] for item in data]
        ids = [item["id"] for item in data]

        video, v_length = self._padding_temporal(video_batch, self.length_video)

        if self.gloss_mode == "concat":
            g_length = torch.tensor(
                [len(item) for item in gloss_batch], dtype=torch.int32
            )
            gloss = torch.concat(gloss_batch)
        elif self.gloss_mode == "padding":
            gloss, g_length = self._padding_temporal(gloss_batch, self.length_gloss)

        video = rearrange(video, "b t c h w -> b c t h w")

        return dict(
            id=ids,
            video=video,
            gloss=gloss,
            video_length=v_length,
            gloss_length=g_length,
            gloss_label=gloss_label,
        )

    @staticmethod
    def _padding_temporal(batch_data: List[torch.Tensor], force_length=None):
        # [t, ....]
        if batch_data is not None:
            if not isinstance(batch_data[0], torch.Tensor):
                raise Exception("data in collate function must be a torch tensor!")
        if force_length is not None:
            # if force temporal length
            t_length = force_length
        else:
            # temporal should be the max
            t_length = max(data.size()[0] for data in batch_data)

        t_lengths_data = []
        ret_data = []
        for data in batch_data:
            t_length_data = data.size()[0]
            t_lengths_data.append(torch.tensor(t_length_data))
            delta_length = t_length - t_length_data
            assert delta_length >= 0
            if delta_length == 0:
                ret_data.append(data)
                continue

            data = torch.transpose(data, 0, -1)
            data = F.pad(data, (0, delta_length), mode="constant", value=0)
            data = torch.transpose(data, 0, -1)
            ret_data.append(data)

        return torch.stack(ret_data), torch.stack(t_lengths_data)
