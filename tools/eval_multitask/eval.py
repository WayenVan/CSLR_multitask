import sys
import numpy as np

sys.path.append("src")
import glob
from pathlib import Path
import pickle
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
import click


@click.option("--work_dir", default="./outputs/evaluate_working_dir")
@click.command()
def main(work_dir: str):
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
            id = data.id
            hyp = data.hyp
            gt = data.gt

        wer = wer_calculation([gt], [hyp])
        print(f"id: {id}\ngt: {gt}\nhyp: {hyp}\nwer:{wer}\n")
        gts.append(gt)
        hyps.append(hyp)
        wers = np.append(wers, wer)

    wer0 = wer_calculation(gts, hyps)
    wer1 = wers.mean()
    print(wer0, wer1)


if __name__ == "__main__":
    main()
