import sys
import numpy as np

sys.path.append("src")
import glob
from pathlib import Path
import pickle
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation
from csi_sign_language.data_utils.ph14.evaluator_sclite import Pheonix14Evaluator
from csi_sign_language.data_utils.ph14.post_process import PostProcess
import click


@click.option("--work_dir", default="./outputs/evaluate_working_dir")
@click.option("--result_dir", default="outputs/train/cache0/validate_data_cache")
@click.option("--data_root", default="dataset/phoenix2014-release")
@click.command()
def main(work_dir: str, result_dir: str, data_root: str):
    results_dir = Path(result_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")
    result_files = glob.glob(str(results_dir / "*.pkl"))
    gts = []
    hyps = []
    ids = []
    wers = np.array([])
    for results in result_files:
        with open(results, "rb") as f:
            data = pickle.load(f)
            id = data["id"]
            hyp = data["hyp"]
            gt = data["gt"]

        wer = wer_calculation([gt], [hyp])
        print(f"id: {id}\ngt: {gt}\nhyp: {hyp}\nwer:{wer}\n")
        gts.append(gt)
        hyps.append(hyp)
        ids.append(id)
        wers = np.append(wers, wer)

    hyps, gts = PostProcess().process(hyps, gts)
    evaluator = Pheonix14Evaluator(
        data_root=data_root, subset="multisigner", mode="dev"
    )
    errors = evaluator.evaluate(ids, hyps, gts, work_dir=work_dir)

    wer = wer_calculation(gts, hyps)

    print(errors)
    print(wer)


if __name__ == "__main__":
    main()
