import abc
from typing import List, Tuple

BatchResult = List[List[str]]


class IPostProcess(abc.ABC):
    @abc.abstractmethod
    def process(
        self, hyp: BatchResult, gt: BatchResult
    ) -> Tuple[BatchResult, BatchResult]:
        pass


class IEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self,
        ids: List[str],
        hyp: BatchResult,
        gt: BatchResult,
        work_dir=None,
    ) -> float:
        """
        evaluator object for evalute the recalled result
        """
        pass
