from typing import List, Tuple
from itertools import groupby
import re

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.data_utils.interface_post_process import IPostProcess
else:
    from ..interface_post_process import IPostProcess


class PostProcess(IPostProcess):
    def __init__(self) -> None:
        super().__init__()

    def process(self, hyps, gts):
        hyp = apply_hypothesis(hyps)
        gt = [apply_groundtruth(item) for item in gts]
        return hyp, gt


def apply_hypothesis(output: List[List[str]], regex=True, merge=True):
    if regex:
        output = [apply_regex(item) for item in output]
    if merge:
        output = [merge_same(item) for item in output]
    return output


def apply_groundtruth(gt):
    # remove repetitions
    gt = [item for item, _ in groupby(gt)]

    # remove __LEFTHAND__ and __EPENTHESIS__ and __EMOTION__
    gt = [item for item in gt if not item == "__LEFTHAND__"]
    gt = [item for item in gt if not item == "__EPENTHESIS__ "]
    gt = [item for item in gt if not item == "__EMOTION__"]

    # remove all -PLUSPLUS suffixes
    gt = [re.sub(r"-PLUSPLUS$", "", item) for item in gt]

    # remove all cl- prefix
    gt = [re.sub(r"^cl-", "", item) for item in gt]
    # remove all loc- prefix
    gt = [re.sub(r"^loc-", "", item) for item in gt]
    # remove RAUM at the end (eg. NORDRAUM -> NORD)
    gt = [re.sub(r"RAUM$", "", item) for item in gt]

    # remove ''
    gt = [item for item in gt if not item == ""]

    # join WIE AUSSEHEN to WIE-AUSSEHEN
    gt = detect_wie(gt)

    # add spelling letters to compounds (A S -> A+S)
    gt = letters_compounds(gt)

    # remove repetitions
    gt = [item for item, _ in groupby(gt)]
    return gt


def apply_regex(output: List[str]):
    """After investigation the shell file, we find that many of these scripts are useless, thus we comment them all"""
    output_s = " ".join(output)

    output_s = re.sub(r"loc-", r"", output_s)
    output_s = re.sub(r"cl-", r"", output_s)
    output_s = re.sub(r"qu-", r"", output_s)
    output_s = re.sub(r"poss-", r"", output_s)
    output_s = re.sub(r"lh-", r"", output_s)
    output_s = re.sub(r"S0NNE", r"SONNE", output_s)
    output_s = re.sub(r"HABEN2", r"HABEN", output_s)

    output_s = re.sub(r"__EMOTION__", r"", output_s)
    output_s = re.sub(r"__PU__", r"", output_s)
    output_s = re.sub(r"__LEFTHAND__", r"", output_s)

    output_s = re.sub(r"WIE AUSSEHEN", r"WIE-AUSSEHEN", output_s)
    output_s = re.sub(r"ZEIGEN ", r"ZEIGEN-BILDSCHIRM ", output_s)
    output_s = re.sub(r"ZEIGEN$", r"ZEIGEN-BILDSCHIRM", output_s)

    output_s = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", output_s)
    output_s = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", output_s)
    output_s = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", output_s)
    output_s = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", output_s)
    output_s = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", output_s)
    output_s = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", output_s)
    output_s = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", output_s)
    output_s = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", output_s)
    output_s = re.sub(r"([ +][A-Z]) ([A-Z]$)", r"\1+\2", output_s)
    output_s = re.sub(r"([A-Z][A-Z])RAUM", r"\1", output_s)
    output_s = re.sub(r"-PLUSPLUS", r"", output_s)

    output_s = re.sub(r"__EMOTION__", r"", output_s)
    output_s = re.sub(r"__PU__", r"", output_s)
    output_s = re.sub(r"__LEFTHAND__", r"", output_s)
    output_s = re.sub(r"__EPENTHESIS__", r"", output_s)

    return output_s.split()


def merge_same(output: List[str]):
    return [x[0] for x in groupby(output)]


def letters_compounds(input):
    state = 0
    output = []
    for item in input:
        if re.match(r"^[A-Z]$", item):
            if state == 0:
                output.append(item)
                state = 1
                continue
            elif state == 1:
                output[-1] = f"{output[-1]}+{item}"
                continue
            else:
                NotImplementedError()
        else:
            output.append(item)
            state = 0
            continue
    return output


def detect_wie(input):
    state = 0
    output = []
    for item in input:
        if item == "WIE":
            output.append(item)
            state = 1
            continue
        if item == "AUSSEHEN":
            if state == 1:
                output[-1] = f"{output[-1]}-{item}"
                state = 0
                continue
            elif state == 0:
                output.append(item)
                continue
        else:
            output.append(item)
            continue
    return output


if __name__ == "__main__":
    print(apply_regex(["S", "S+H", "C", "E", "loc-HOHOHO", "__ON__", "__EMOTION__"]))
