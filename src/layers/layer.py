import cupy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    '''
    Abstract class representing a layer of the neural network. It exposes the forward and backward methods.
    '''
    @abstractmethod
    def initialize(self, layer_idx: int, num_layers: int) -> None:
        pass
    
    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, wrong_samples: np.ndarray, next_bwd_out: np.ndarray, algo_layer: str, acc: float, prob_reinforcement: float, freeze_last: bool = False) -> np.ndarray:
        pass