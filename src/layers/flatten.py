import numpy as np

from layers import Layer

class FlattenLayer(Layer):
    '''
    Class that implements a flatten layer.
    '''
    def initialize(self, layer_idx: int, num_layers: int) -> None:
        '''
        Initialize the flatten layer.
        
        Parameters
        ----------
        layer_idx : int
            Index of the layer in the network.
        num_layers : int
            Total number of layers in the network.
        '''
        self.last = layer_idx == num_layers - 1
        
    def forward(self, input: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the FlattenLayer. It flattens the input in a 1D array.
        
        Parameters
        ----------
        input : numpy.ndarray
            Input multidimensional vector.
            
        Returns
        -------
        numpy.ndarray
            Flattened input.
        '''
        # Get input dimension
        dimension = input.shape
        
        # Reshape the input based on its dimensions
        flatten_input = input.reshape(dimension[0], np.prod(dimension[1:]))
        
        return flatten_input
    
    def backward(self, wrong_samples: np.ndarray, next_bwd_out: np.ndarray, algo_layer: str, acc: float, prob_reinforcement: float, freeze_last: bool = False) -> np.ndarray:
        pass