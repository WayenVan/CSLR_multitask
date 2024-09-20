import sys
import numpy as np

sys.path.append("src")
import glob
from pathlib import Path
import pickle
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
from csi_sign_language.data.datamodule.ph14 import Ph14DataModule
import click

work_dir = "./outputs/evaluate_working_dir"
ph14_root = "dataset/phoenix2014-release"
ph14_lmdb_root = "dataset/preprocessed/ph14_lmdb"


def main():
    dm = Ph14DataModule(ph14_lmdb_root, 1, 4, False)
    results_dir = Path(work_dir) / "test_result"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    result_files = glob.glob(str(results_dir / "*.pkl"))
    gts = []
    hyps = []
    wers = np.array([])
    for results in result_files:
        with open(results, "rb") as f:
            data = pickle.load(f)
            id = data["id"]
            hyp = data["hyp"]
            gt = data["gt"]
            # [t c]
            out = data["out"]

        wer = wer_calculation([gt], [hyp])
        vocab = dm.get_vocab()
        if wer > 40.0:
            out = out.argmax(-1)
            print(id)
            print(vocab(gt))
            # print(out.shape)
            print(out)
            print(wer)
            print("--------")

        gts.append(gt)
        hyps.append(hyp)
        wers = np.append(wers, wer)

    wer0 = wer_calculation(gts, hyps)
    wer1 = wers.mean()
    print(wer0, wer1)


if __name__ == "__main__":
    main()
