import cupy as cp

from layers import Layer

class FCLayer(Layer):
    '''
    Class that implements a fully connected layer.
    '''
    def __init__(self, input_dim: int, output_dim: int, group_size: int = 0, freeze: bool = False, weight_clip: int = None, grouping_layer: bool = False) -> None:
        '''
        Class that implements a fully connected layer.
        
        Parameters
        ----------
        input_dim : int
            Input dimension of the layer.
        output_dim : int
            Output dimension of the layer.
        group_size : int (default = 0)
            The number of groups used by the updating rule.
        freeze : bool, optional (default = False)
            If True, the layer is frozen and the weights are not updated during the backward pass.
        weight_clip : int, optional (default = None)
            If not None, the weights are clipped to the range [-weight_clip, weight_clip]
        grouping_layer : bool, optional (default = False)
            If True, the layer is a grouping layer and the weights are all 0s and 1s in groups (Baldassi et al. work).
        '''
        self.input_dim, self.output_dim = input_dim, output_dim
        self.freeze = freeze
        self.grouping_layer = grouping_layer
        self.group_size = group_size
        self.weight_clip = weight_clip
        self.weights = None

    def initialize(self, layer_idx: int, num_layers: int) -> None:
        '''
        Initialize the layer. It initializes the weights of the layer (randomly or as grouping layer).
        
        Parameters
        ----------
        layer_idx : int
            Index of the layer in the network.
        num_layers : int
            Total number of layers in the network.
        max_dimension : int
            Dimension of the largest layer of the network.
        free_memory : bool
            If True, free the GPU memory after the training of the layer.
        '''
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.last = layer_idx == num_layers - 1
        
        # If grouping_layer, weights are all 0s and 1s in groups (Baldassi et al. work)
        if self.grouping_layer:
            n_group = self.input_dim // self.output_dim
            indices = cp.arange(self.input_dim) // n_group
            self.weights = (indices[:, None] == cp.arange(self.output_dim)).astype(int)
        # Otherwise, weights are randomly initialized with a normal distribution
        else:
            self.weights = cp.random.normal(0, 64, size=(self.input_dim, self.output_dim)).astype(int)
        
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        '''
        Forward pass of the FCLayer. It computes the pre activation and the activation of the layer.
        
        Parameters
        ----------
        input : numpy.ndarray
            Input array of the layer.
        
        Returns
        -------
        numpy.ndarray
            Activation of the layer if not the last layer, otherwise the pre activation.
        '''
        # Save the input for the backward pass
        self.input = input

        # Compute binary weights
        binary_w = cp.sign(self.weights)
        
        # Compute the pre activation
        self.pre_activations = (self.input @ binary_w)
        
        # Compute the activation (sign function)
        self.activations = cp.sign(self.pre_activations)
        
        # Return the activations if not the last layer
        if not self.last:
            return self.activations
        # Otherwise return the pre activations
        else:
            return self.pre_activations
    
    def backward(self, wrong_samples: cp.ndarray, next_bwd_out: cp.ndarray, algo_layer: str, acc: float, prob_reinforcement: float, freeze_last: bool = False) -> cp.ndarray:
        '''
        Backward pass of the FCLayer. It updates the weights of the layer according to the chosen algorithm.
        
        Parameters
        ----------
        wrong_samples : numpy.ndarray
            Array of booleans that indicates the wrong samples in the batch.
        next_bwd_out : numpy.ndarray
            Output of the next layer backward pass. If last layer, it is the one hot encoding of the labels.
        algo_layer : int
            Learning algorithm at layer level (Update all perceptrons, Update only one perceptron).
        acc : float, optional (default = 0.0)
            Running accuracy of the network.
        prob_reinforcement : float
            Probability of reinforcement.
        freeze_last : bool, optional (default = False)
            If True, the last layer is frozen and the weights are not updated during the backward pass.
            
        Returns
        -------
        numpy.ndarray
            If last layer, the weights associated to the correct label which represent the desired activations. Otherwise, the sum of the weights associated to the correct perceptrons.
        '''
        # Rescale the probability of reinforcement
        prob_reinforcement = prob_reinforcement * cp.sqrt(1.0-acc/100.0) * cp.sqrt(2/(cp.pi*self.input_dim))

        # If last layer, update the weights if layer is not frozen and return the weights associated to the wrong perceptrons
        if self.last and freeze_last:
            # If binary classification, return the weights multiplied by the next_bwd_out
            if next_bwd_out.shape[1] == 1:
                return cp.sign(next_bwd_out * self.weights.T)
            
            # Return the weights associated to the correct perceptrons
            correct_labels = next_bwd_out.argmax(axis=1)
            return cp.sign(self.weights.T[correct_labels])
        
        # If grouping layer, propagate the weights to the wrong perceptrons
        elif self.grouping_layer:
            # Find the correct/wrong perceptrons
            perceptron_errors = (self.activations[wrong_samples] * next_bwd_out)
            
            # Set to 0 the weights associated to the correct perceptrons
            perceptron_errors[perceptron_errors >= 0] = 0
            
            # Set to the desired output the weights associated to the wrong perceptrons
            perceptron_errors[perceptron_errors < 0] = next_bwd_out[perceptron_errors < 0]
            
            # Return the desired activations
            perceptron_errors = cp.expand_dims(perceptron_errors, axis=1)
            return cp.sum(self.weights * perceptron_errors, axis=2)
        
        # Otherwise, update the weights if layer is not frozen
        elif not self.freeze:
            return self.update_weights(wrong_samples, next_bwd_out, algo_layer, prob_reinforcement)
    
    def update_weights(self, wrong_samples: cp.ndarray, next_bwd_out: cp.ndarray, algo_layer: str, prob_reinforcement: float) -> None:
        '''
        Layer level algorithm that updates the perceptrons of the layer if not frozen according to the chosen layer level (all or one) and perceptron level algorithms (sbpi or cp+r).
        
        Parameters
        ----------
        wrong_samples : numpy.ndarray
            Array of booleans that indicates the wrong samples in the batch.
        next_bwd_out : numpy.ndarray
            Output of the next layer backward pass. If last layer, it is the one hot encoding of the labels.
        algo_layer : int
            Learning algorithm at layer level (our, baldassi).
        prob_reinforcement : float
            Probability of reinforcement.
        '''
        # Transpose the weights and compute the sign of the input
        transposed_weights = self.weights.T
        input_sign = cp.sign(self.input).astype(int)
        
        # Create the mask of the perceptrons to update according to the chosen algorithm
        match algo_layer:
            case "our":
                # Find the correct/wrong perceptrons
                perceptron_errors = (self.pre_activations[wrong_samples] * next_bwd_out)
                    
                # Find the perceptron which makes the lowest error in each of the subgroups
                # Step 1: Reshape the perceptron_errors array to have the same number of rows but with group_size columns
                perceptron_errors = perceptron_errors.reshape(-1, self.output_dim//self.group_size, self.group_size)
                
                # Step 2: Find the negative values
                negative_values = perceptron_errors < 0

                # Step 3: Calculate the absolute values of perceptron_errors, but keep only negatives as valid
                absolute_negatives = cp.where(negative_values, cp.abs(perceptron_errors), cp.inf)

                # Step 4: Get the rank of each negative value in its row
                ranks = cp.argsort(cp.argsort(absolute_negatives, axis=2), axis=2)

                # Step 5: Mark the smallest negative as True
                mask = (ranks < 1) & negative_values
                
                # Flatten the mask for each subgroup
                mask = mask.reshape(mask.shape[0], -1)
                
            case "baldassi":
                # Find the correct/wrong perceptrons
                perceptron_errors = (self.pre_activations[wrong_samples] * next_bwd_out)
                
                # Find the perceptron which makes the lowest error in each of the subgroups (of size group_size)
                # Step 1: Reshape the perceptron_errors array to have the same number of rows but with group_size columns
                perceptron_errors = perceptron_errors.reshape(-1, self.output_dim//self.group_size, self.group_size)
                
                # Step 2: Find the perceptron which makes the lowest error in each of the subgroups
                idx_to_update = cp.argmax(cp.where(perceptron_errors < 0, perceptron_errors, -cp.inf), axis=2)                
                
                # Step 3: Compute the indexes of the perceptrons to update in the original perceptron_errors array (multiply grouping factor per column index of idx_to_update)
                idx_to_update += cp.arange(0, self.output_dim, self.group_size)

                # Step 4: Exclude the indexes that correspond to groups with all 0s in perceptron_errors (set them to -1)
                idx_to_update = cp.where(cp.any(perceptron_errors < 0, axis=2), idx_to_update, -1)
                                
                # Step 5: Create the mask of the perceptrons to update
                mask = cp.zeros(next_bwd_out.shape).astype(cp.int32)
                row_indices, _ = cp.indices(idx_to_update.shape)
                valid_mask = idx_to_update >= 0
                mask[row_indices[valid_mask], idx_to_update[valid_mask]] = True
                
            case _:
                raise ValueError(f"Invalid value for algo_layer: {algo_layer}")
                
        # Update the weights associated to the wrong perceptrons
        transposed_weights += (2 * (input_sign[wrong_samples]).T @ (next_bwd_out * mask)).T

        # Compute the reinforcement of the weights according to the chosen algorithm
        reinforced_weights = self.compute_reinforcement(transposed_weights, prob_reinforcement)
        
        # Save the updated clipped weights
        if self.weight_clip is None:
            # Save the updated weights
            self.weights = reinforced_weights.T
        else:
            self.weights = cp.clip(reinforced_weights.T, -self.weight_clip, self.weight_clip)
        
        new_next_bwd_out = cp.zeros(next_bwd_out.shape).astype(cp.int32)
        if self.last:
            # Return the weights associated to the correct perceptrons
            correct_labels = next_bwd_out.argmax(axis=1)
            new_next_bwd_out = cp.sign(self.weights.T[correct_labels])

        return new_next_bwd_out
    
    def compute_reinforcement(self, transposed_weights: cp.ndarray, prob_reinforcement: float) -> cp.ndarray:
        '''
        Compute the reinforcement of all the weights with probability on the output perceptrons
        
        Parameters
        ----------
        transposed_weights : numpy.ndarray
            Transposed weights of the layer.
        prob_reinforcement : float
            Probability of reinforcement.
        '''
        # Define a random array for reinforcement on the output perceptrons
        random_array = cp.random.random(transposed_weights.shape) < prob_reinforcement

        # Reinforce the weights according to the random array
        transposed_weights += 2 * cp.sign(transposed_weights) * random_array

        return transposed_weights