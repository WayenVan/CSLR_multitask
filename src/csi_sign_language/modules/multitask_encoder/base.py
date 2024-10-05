from abc import ABC, abstractmethod
from torch import nn

from typing import List, Tuple


class VisualBackbone(ABC, nn.Module):
    @abstractmethod
    def get_input_size() -> Tuple[int, int]:
        """
        return the required input size of the visual backbone, (H, W)
        """
        pass

    @abstractmethod
    def get_output_feats_size() -> List[Tuple[int, int]]:
        """
        return  a list of output feats, each corrresponding to a different stage, the element is (H, W)
        """
        pass

    @abstractmethod
    def get_output_dims() -> List[int]:
        """
        @return a list of output dims, each corrresponding to a different stage
        """
        pass
