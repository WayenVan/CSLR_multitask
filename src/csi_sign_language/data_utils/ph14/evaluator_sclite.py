import os
import numpy as np
from typing import List
import shutil


if __name__ == "__main__":
    pass
else:
    from ..utils.misc import add_exe_mode, glosses2ctm
    from ..base import IEvaluator


class Pheonix14Evaluator(IEvaluator):
    def __init__(self, data_root, subset, mode) -> None:
        if subset == "multisigner":
            self.root = os.path.join(data_root, "phoenix-2014-multisigner")
        elif subset == "si5":
            self.root = os.path.join(data_root, "phoenix-2014-signerindependent-SI5")
        else:
            raise NotImplementedError()

        self.eval_root = os.path.join(self.root, "evaluation")

        self.merge_file = os.path.join(self.eval_root, "mergectmstm.py")
        self.eval_shell = os.path.join(self.eval_root, "evaluatePhoenix2014.sh")
        self.dev_stm = os.path.join(self.eval_root, "phoenix2014-groundtruth-dev.stm")
        self.test_stm = os.path.join(self.eval_root, "phoenix2014-groundtruth-test.stm")
        self.mode = mode

    def evaluate(
        self, ids: List[str], hyp: List[List[str]], gt: List[List[str]], work_dir=None
    ):
        # NOTE: actually the ground truth is not used, because we already have the .stm file
        if work_dir is None:
            raise RuntimeError(
                "the Phoenix14Evaluator need a work_dir to store the temp files"
            )
        if self.mode not in ("dev", "test"):
            raise NotImplementedError()

        os.makedirs(work_dir, exist_ok=True)
        self.prepare_resources(work_dir)
        glosses2ctm(ids, hyp, os.path.join(work_dir, "hyp.ctm"))
        env = "export PATH=./:$PATH;"
        cmd1 = f"cd {work_dir};"
        cmd2 = f"sh ./eval.sh hyp.ctm {self.mode}"
        if os.system(env + cmd1 + cmd2) != 0:
            raise Exception("sclit cmd runing failed")

        result_file = os.path.join(work_dir, "out.hyp.ctm.sys")

        with open(result_file, "r") as fid:
            for line in fid:
                line = line.strip()
                if "Sum/Avg" in line:
                    result = line
                    break

        tmp_err = result.split("|")[3].split()
        subs, inse, dele, wer = tmp_err[1], tmp_err[3], tmp_err[2], tmp_err[4]
        subs, inse, dele, wer = float(subs), float(inse), float(dele), float(wer)
        # errs = [wer, subs, inse, dele]
        return wer

    def prepare_resources(self, work_dir):
        eval_shell = os.path.join(work_dir, "eval.sh")
        merge_file = os.path.join(work_dir, "mergectmstm.py")
        dev_stm = os.path.join(work_dir, "phoenix2014-groundtruth-dev.stm")
        test_stm = os.path.join(work_dir, "phoenix2014-groundtruth-test.stm")
        shutil.copyfile(self.eval_shell, eval_shell)
        shutil.copyfile(self.merge_file, merge_file)
        shutil.copyfile(self.dev_stm, dev_stm)
        shutil.copyfile(self.test_stm, test_stm)
        add_exe_mode(merge_file)
