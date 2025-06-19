import argparse
import cupy as cp
import numpy as np

from layers import Layer

MAX_BATCH_SIZE = 1000

class SubNetwork:
    '''
    Class representing a subnetwork of the main super network. 
    '''
    def __init__(self):
        '''
        Class representing a subnetwork of the main super network. It contains a list of layers, of which the first one is the forward layer of the super network.
        '''
        self.layers: list[Layer] = []
        self.freeze = False

    def add(self, layer: Layer) -> None:
        '''
        Add a layer to the subnetwork.
        
        Parameters
        ----------
        layer : Layer
            Layer to be added.
        '''
        self.layers.append(layer)
    
    def initialize(self, net_idx: int, num_nets: int) -> None:
        '''
        Initialize the layers of the subnetwork.
        
        Parameters
        ----------
        net_idx : int
            Index of the subnetwork in the super network.
        num_nets : int
            Total number of subnetworks in the super network.
        '''
        self.first = net_idx == 0
        self.last = net_idx == num_nets - 1

        # If all FC layers are frozen, freeze the subnetwork
        if all([layer.freeze for layer in self.layers if hasattr(layer, "weights")]):
            self.freeze = True
        
        # Initialize the layers
        for idx, layer in enumerate(self.layers):
            layer.initialize(idx, len(self.layers))
    
    def test(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        '''
        Test the sub network on the given dataset.
        
        Parameters
        ----------
        x_set : numpy.ndarray
            Input dataset to test.
        y_set : numpy.ndarray
            Target labels of the dataset.
            
        Returns
        -------
        float
            Accuracy of the sub network on the given dataset.
        '''
        # Predict the output
        pred = self.predict(x_set)

        # If the target is one-hot encoded and it is not binary classification, get the index of the maximum value
        if y_set.ndim > 1 and self.layers[-1].output_dim > 1:
            target = np.argmax(y_set, axis=1)
        else:
            target = y_set
            
        # Compute the accuracy
        acc = 100.0 * np.sum(pred == target) / len(x_set)
        return acc.item()
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        '''
        Predict the output of the sub network on the given input data.
        
        Parameters
        ----------
        input_data : numpy.ndarray
            Input data on which to predict the output.
            
        Returns
        -------
        numpy.ndarray
            Predicted output of the sun network.
        '''
        # Perform the forward pass of the sub network in batch mode
        num_batches = len(input_data)//MAX_BATCH_SIZE + 1 if len(input_data)%MAX_BATCH_SIZE != 0 else len(input_data)//MAX_BATCH_SIZE
        for batch in range(num_batches):
            # Compute the starting and ending index of the batch
            idx_start_batch = batch * MAX_BATCH_SIZE
            idx_end_batch = idx_start_batch + MAX_BATCH_SIZE
            if idx_start_batch >= len(input_data):
                break
            elif idx_end_batch > len(input_data):
                idx_end_batch = len(input_data)

            output = cp.asarray(input_data[idx_start_batch:idx_end_batch])
            
            for layer in self.layers:
                output = layer.forward(output)
                                    
            if batch == 0:
                outputs = cp.asnumpy(output)
            else:
                outputs = np.concatenate((outputs, cp.asnumpy(output)))
        
        # Return the predicted output
        if outputs.shape[1] == 1:
            # If binary classification, return the sign of the output
            return np.sign(outputs)
        else:
            # If multiclass classification, return the index of the maximum value of the output
            return np.argmax(outputs, axis=1)
    
    def fit(self, x_batch: np.ndarray, y_batch: np.ndarray, args: argparse.Namespace, acc: float) -> np.ndarray:
        '''
        Fit the subnetwork on the given batch. It performs a forward pass and a backward pass for each layer of the subnetwork.
        
        Parameters
        ----------
        x_batch : numpy.ndarray
            Input batch of data.
        y_batch : numpy.ndarray
            Target batch of data.
        args : argparse.Namespace
            Dictionary containing the arguments passed to the program.
        acc : float
            Accuracy of the previous epoch.
            
        Returns
        -------
        numpy.ndarray
            The output of the last layer of the subnetwork if it is the last subnetwork, the activations of the first layer of the subnetwork otherwise.
        '''
        # Forward propagation
        for layer in self.layers:
            x_batch = layer.forward(x_batch)
        fwd_out = x_batch

        # Backward propagation if the subnetwork is not frozen
        if not self.freeze:
            # If binary classification, compute the mask of wrong samples and the robustness parameter accordingly
            if self.layers[-1].output_dim == 1:
                wrong_samples = np.reshape(np.sign(y_batch) != np.sign(fwd_out), -1)                
                
                # If the robustness parameter is not 0, modify the mask of wrong samples accordingly
                if args.rob != 0.0:
                    # If the absolute value of fwd_out is less than the layer input dimension times the robustness parameter, consider the sample as wrong
                    row_indices = np.argwhere(np.abs(fwd_out) < int(self.layers[-1].input_dim * args.rob))[:, 0]
                    wrong_samples[row_indices] = True
            else:
                # Compute the mask of wrong samples
                wrong_samples = np.argmax(y_batch, axis=1) != np.argmax(fwd_out, axis=1)
                
                # If the robustness parameter is not 0, modify the mask of wrong samples accordingly
                if args.rob != 0.0:
                    # Compute the difference between the two largest values of the output activations
                    two_largest = np.partition(fwd_out, -2, axis=1)[:, -2:]
                    max_diff = np.diff(two_largest, axis=1)

                    # If the difference is less than the layer input dimension times the robustness parameter, consider the sample as wrong
                    row_indices = np.argwhere(max_diff < int(self.layers[-1].input_dim * args.rob))[:, 0]
                    wrong_samples[row_indices] = True
            
            # If all wrong_samples are False, do not update the weights
            if np.any(wrong_samples):
                bwd_out = y_batch[wrong_samples]
                for layer in reversed(self.layers):
                    bwd_out = layer.backward(wrong_samples, bwd_out, args.algo_layer, acc, args.prob_reinforcement, args.freeze_last)

        # Return the output of the last layer if it is the last subnetwork, the activations of the first layer otherwise
        if self.last:
            return fwd_out
        return self.layers[1].activations if self.first else self.layers[0].activations
    
    def __repr__(self) -> str:
        '''
        Return a string representation of the subnetwork.
        '''
        layers = ""
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                layers += f"{layer.__class__.__name__}(({layer.input_dim}, {layer.output_dim}), freeze: {layer.freeze})"
                layers += f" -> " if not layer.last else ""
            else:
                layers += f"{layer.__class__.__name__} -> " if not layer.last else f"{layer.__class__.__name__}"       
        return layers