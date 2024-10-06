from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
from typing import List, Any, Union, NamedTuple
from einops import rearrange
from ..utils.decode import CTCBeamDecoder
from ..utils.misc import is_namedtuple_instance, clean_folder
from hydra.utils import instantiate

import lightning as L
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from csi_sign_language.data_utils.ph14.post_process import apply_hypothesis
from csi_sign_language.modules.losses.loss import VACLoss as _VACLoss
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
import pickle
import glob
import numpy as np

from ..data_utils.base import IEvaluator


class SLRModel(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        vocab,
        ctc_search_type="beam",
        file_logger=None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=["cfg", "file_logger"])

        self.cfg = cfg
        self.data_excluded = getattr(cfg, "data_excluded", [])
        self.backbone: nn.Module = instantiate(cfg.model)

        self.loss = instantiate(cfg.loss)

        self.vocab = vocab
        self.decoder = CTCBeamDecoder(vocab, blank_id=0, log_probs_input=True)

        self.train_ids_epoch = []
        self.val_ids_epoch = []

    @torch.no_grad()
    def _ctc_decode(self, out, length):
        # [t n c]
        # return list(list(string))
        y_predict = torch.nn.functional.log_softmax(out, -1).detach().cpu()
        results = self.decoder(y_predict, length)
        return results

    @staticmethod
    def _gloss2sentence(gloss: List[List[str]]):
        return [" ".join(g) for g in gloss]

    @staticmethod
    def _extract_batch(batch):
        video = batch["video"]
        gloss = batch["gloss"]
        video_length = batch["video_length"]
        gloss_length = batch["gloss_length"]
        gloss_gt = batch["gloss_label"]
        id = batch["id"]
        return id, video, gloss, video_length, gloss_length, gloss_gt

    @staticmethod
    def check_gradients(module, grad_input, grad_output):
        for grad in grad_input:
            if grad is not None:
                if torch.isnan(grad).any():
                    print("graident is nan", file=sys.stderr)
                if torch.isinf(grad).any():
                    print(grad)
                    print("graident is inf", file=sys.stderr)

    def set_post_process(self, fn):
        self.post_process = fn

    def set_work_dir(self, work_dir: str):
        self.work_dir = Path(work_dir)

    def forward(self, x, t_length) -> Any:
        outputs = self.backbone(x, t_length)
        hyp = self._ctc_decode(outputs.out, outputs.t_length)[0]
        return outputs, hyp

    def predict_step(self, batch, batch_id) -> Any:
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(
            batch
        )
        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            hyp = self._ctc_decode(outputs.out, outputs.t_length)[0]
        # [(id, hyp, gloss_gt), ...]
        return list(zip(id, hyp, gloss_gt))

    def on_train_epoch_start(self):
        self.train_ids_epoch.clear()

    def training_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(
            batch
        )
        B = len(id)

        try:
            outputs = self.backbone(video, video_length)
            loss_output = self.loss(outputs, video, video_length, gloss, gloss_length)
            # NOTE: loss could be namedtuple or a value, when neamedtuple, log all losses provided inside
            if is_namedtuple_instance(loss_output):
                loss = loss_output.out
                for key, value in loss_output._asdict().items():
                    if value is not None:
                        self.log(
                            f"train_loss_{key}",
                            value,
                            on_epoch=True,
                            on_step=True,
                            prog_bar=True,
                            sync_dist=False,
                            batch_size=B,
                        )
            else:
                loss = loss_output
                self.log(
                    "train_loss",
                    loss,
                    on_epoch=True,
                    on_step=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=B,
                )
        except torch.cuda.OutOfMemoryError as e:
            print(f"cuda out of memory! the t_length is {video.size(2)}")
            raise e

        if torch.isnan(loss) or torch.isinf(loss):
            self.print(
                f"find nan, data id={id}, output length={outputs.t_length.cpu().numpy()}, label_length={gloss_length.cpu().numpy()}",
                file=sys.stderr,
            )
            raise ValueError("find nan in loss")

        hyp = self._ctc_decode(outputs.out.detach(), outputs.t_length.detach())[0]

        self.log(
            "train_wer",
            wer_calculation(gloss_gt, hyp),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.train_ids_epoch += id
        return loss

    def set_evaluator(self, evaluator: IEvaluator):
        self.evaluator = evaluator

    def on_validation_start(self) -> None:
        # check if evaluator has been  set
        if not hasattr(self, "evaluator"):
            raise RuntimeError(
                "missing evaluator, please use self.set_evaluator to set it"
            )
        if not hasattr(self, "work_dir"):
            raise RuntimeError(
                "missing working dir, please use self.set_work_dir to set it"
            )
        # setup path variable
        self.validation_data_cache = self.work_dir / "validation_cache"
        self.validation_data_cache.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_start(self) -> None:
        # clean previous result validatation_work_dir
        clean_folder(str(self.validation_data_cache))
        self.val_ids_epoch.clear()

    def validation_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(
            batch
        )
        B = len(id)

        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            loss_output = self.loss(outputs, video, video_length, gloss, gloss_length)
            if is_namedtuple_instance(loss_output):
                loss = loss_output.out
                for key, value in loss_output._asdict().items():
                    if value is not None:
                        self.log(
                            f"val_loss_{key}",
                            value,
                            on_epoch=True,
                            on_step=False,
                            prog_bar=True,
                            sync_dist=True,
                            batch_size=B,
                        )
            else:
                loss = loss_output
                self.log(
                    "val_loss",
                    loss,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=B,
                )

        hyp = self._ctc_decode(outputs.out, outputs.t_length)[0]
        # if self.post_process:
        #     hyp, gt = self.post_process.process(hyp, gloss_gt)
        # else:
        #     raise NotImplementedError()

        self.val_ids_epoch += id
        # return all possible infomation about the result
        for b in range(len(id)):
            result = dict(
                id=id[b],
                hyp=hyp[b],
                gt=gloss_gt[b],
                out=outputs.out.cpu().numpy()[:, b],
                t_length=outputs.t_length.cpu().numpy()[b],
            )
            # print(outputs.out.shape[0])
            with (self.validation_data_cache / f"{result['id']}.pkl").open("wb") as f:
                pickle.dump(result, f)
        if self.trainer.is_last_batch:
            self.log("test", 1)
            self.print("last batch")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.global_rank != 0:
            return
        validation_results = glob.glob(str(self.validation_data_cache / "*.pkl"))
        gts = []
        hyps = []
        ids = []
        wers = np.array([])
        for result in validation_results:
            with open(result, "rb") as f:
                data = pickle.load(f)
                id = data["id"]
                hyp = data["hyp"]
                gt = data["gt"]

            wer = wer_calculation([gt], [hyp])
            gts.append(gt)
            hyps.append(hyp)
            ids.append(id)
            wers = np.append(wers, wer)
        self.log(
            "val_wer",
            wers.mean(),
            on_epoch=True,
            on_step=False,
            rank_zero_only=True,
        )
        # native wer
        wer_native = self.evaluator.evaluate(
            ids, hyps, gts, work_dir=self.work_dir / "validate_work_dir"
        )
        self.log(
            "val_wer_native",
            wer_native,
            on_epoch=True,
            on_step=False,
            rank_zero_only=True,
        )

    def set_test_working_dir(self, dir: str):
        self.test_working_dir = Path(dir)

    def on_test_start(self) -> None:
        # check value
        if not hasattr(self, "test_working_dir"):
            raise RuntimeError(
                "working dir is not set, plese use self.set_test_working_dir to set it"
            )
        clean_folder(str(self.validation_data_cache))
        # make sure the test cache is emptye

    def test_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(
            batch
        )

        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            loss = self.loss(outputs, video, video_length, gloss, gloss_length)
            hyp, beam_result, beam_score, timesteps, out_seq_len = self._ctc_decode(
                outputs.out, outputs.t_length
            )

        if self.post_process:
            hyp, gt = self.post_process.process(hyp, gloss_gt)
        else:
            raise NotImplementedError()

        # return all possible infomation about the result
        for b in range(len(id)):
            result = dict(
                id=id[b],
                hyp=hyp[b],
                gt=gt[b],
                out=outputs.out.cpu().numpy()[:, b],
                t_length=outputs.t_length.cpu().numpy()[b],
                beam_result=beam_result.cpu().numpy()[b],
                beam_score=beam_score.cpu().numpy()[b],
                timesteps=timesteps.cpu().numpy()[b],
                out_seq_len=out_seq_len.cpu().numpy()[b],
            )
            # print(outputs.out.shape[0])
            test_result_output_folder = self.test_working_dir / "test_result"
            test_result_output_folder.mkdir(parents=True, exist_ok=True)
            with (test_result_output_folder / f"{result['id']}.pkl").open("wb") as f:
                pickle.dump(result, f)

    def configure_optimizers(self):
        opt: Optimizer = instantiate(
            self.cfg.optimizer,
            filter(lambda p: p.requires_grad, self.backbone.parameters()),
        )
        scheduler = instantiate(self.cfg.lr_scheduler, opt)
        return {"optimizer": opt, "lr_scheduler": scheduler}


class VACLoss(nn.Module):
    def __init__(self, weights, temp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.loss = _VACLoss(weights, temp)

    def forward(self, outputs, input, input_length, target, target_length):
        if self.weights[2] > 0.0 or self.weights[1] > 0.0:
            conv_out = outputs.neck_out.out
            conv_length = outputs.neck_out.t_length
        else:
            conv_out = None
            conv_length = None

        seq_out = outputs.out
        t_length = outputs.t_length

        return self.loss(
            conv_out, conv_length, seq_out, t_length, target, target_length
        )
