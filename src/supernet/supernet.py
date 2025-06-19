import os
import copy
import time
import textwrap
import argparse
import cupy as cp
import numpy as np

from subnet import SubNetwork
from layers import FlattenLayer, FCLayer
from dataset.utils import shuffle_dataset

BARS = 50
PRINT_INTERVAL = 15
MAX_BATCH_SIZE = 1000

class SuperNetwork:
    '''
    Class representing the super network.
    '''
    def __init__(self, layers: str, input_dim: int, output_dim: int, freeze_first: bool, freeze_last: bool, group_size: int, weight_clip: int) -> None:
        '''
        Class representing the super network. It contains a list of subnetworks built from the arguments passed to the program.

        Parameters
        ----------
        layers : str
            String containing the dimensions of the layers of the subnetworks.
        input_dim : int
            Input dimension of the super network.
        output_dim : int
            Output dimension of the super network (number of classes).
        freeze_first : bool
            If True, the first layer of the first subnetwork is frozen.
        freeze_last : bool
            If True, the last layer of the subnetworks is frozen.
        group_size : int
            The number of groups used by the updating rule.
        weight_clip : int
            If not None, the weights of the layers are clipped to the given value.
        '''
        self.subnets: list[SubNetwork] = []

        # Create the subnetworks from the arguments
        layers_list = layers.split("_")
        for idx, layer in enumerate(layers_list):
            try:
                K1, K2 = map(int, layer.split("-"))
            except ValueError:
                K1, K2 = int(layer), 0

            # If first subnetwork, create a subnetwork with the Flatten layer
            if idx == 0:
                freeze_last = True if freeze_first else freeze_last
                net = self.create_subnet(input_dim, K1, K2, output_dim, group_size, weight_clip, flatten=True, freeze_first=freeze_first, freeze_last=freeze_last)
            else:
                net = self.create_subnet(K1prev, K1, K2, output_dim, group_size, weight_clip, freeze_last=freeze_last)

            self.subnets.append(net)
            K1prev = K1

        # Initialize the subnetworks
        for idx, net in enumerate(self.subnets):
            net.initialize(idx, len(self.subnets))

    def create_subnet(self, input_dim: int, hidden1_dim: int, hidden2_dim: int, output_dim: int, group_size: int, weight_clip: int, flatten: bool = False, freeze_first: bool = False, freeze_last: bool = True) -> tuple[SubNetwork, int]:
        '''
        Create a subnetwork with the given parameters.

        Parameters
        ----------
        input_dim : int
            Input dimension of the subnetwork.
        hidden1_dim : int
            Dimension of the first hidden layer.
        hidden2_dim : int
            Dimension of the second hidden layer (grouping layer in Baldassi et al.).
        output_dim : int
            Output dimension of the subnetwork (number of classes).
        group_size : int, optional (default = 0)
            The number of groups used by the updating rule.
        weight_clip : int
            If not None, the weights of the layers are clipped to the given value.
        flatten : bool (default = False)
            If True, the Flatten layer is added to the subnetwork.
        freeze_first : bool (default = False)
            If True, the first layer of the subnetwork is frozen.
        freeze_last : bool (default = True)
            If True, the last layer of the subnetworks is frozen.

        Returns
        -------
        tuple[SubNetwork, int]
            Tuple containing the subnetwork.
        '''
        # Check that the dimensions are correct (hidden2_dim is used for the grouping layer)
        if hidden2_dim != 0:
            assert(hidden1_dim % hidden2_dim == 0)
        if not freeze_first:
            assert(hidden1_dim % group_size == 0)
         
        # Build the subnetwork
        net = SubNetwork()
        if flatten:
            net.add(FlattenLayer())
        net.add(FCLayer(input_dim, hidden1_dim, group_size=group_size, freeze=freeze_first, weight_clip=weight_clip))

        if hidden2_dim != 0:
            net.layers[-1].group_size = hidden1_dim//hidden2_dim
            net.add(FCLayer(hidden1_dim, hidden2_dim, freeze=True, grouping_layer=True))
            net.add(FCLayer(hidden2_dim, output_dim, freeze=True))
        else:
            net.add(FCLayer(hidden1_dim, output_dim, group_size=output_dim, freeze=freeze_last, weight_clip=weight_clip))

        # Return the subnetwork
        return net

    def test(self, x_set: np.ndarray, y_set: np.ndarray) -> float:
        '''
        Test the super network on the given dataset.

        Parameters
        ----------
        x_set : numpy.ndarray
            Input dataset to test.
        y_set : numpy.ndarray
            Target labels of the dataset.

        Returns
        -------
        float
            Accuracy of the super network on the given dataset.
        '''
        # Predict the output
        pred = self.predict(x_set)

        # If the target is one-hot encoded and it is not binary classification, get the index of the maximum value
        if y_set.ndim > 1 and self.subnets[-1].layers[-1].output_dim > 1:
            target = np.argmax(y_set, axis=1)
        else:
            target = y_set
            
        # Compute the accuracy
        acc = 100.0 * np.sum(pred == target) / len(x_set)
        return acc.item()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        '''
        Predict the output of the super network on the given input data.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data on which to predict the output.

        Returns
        -------
        numpy.ndarray
            Predicted output of the super network.
        '''
        # Perform the forward pass of the super network in batch mode
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
            for net in self.subnets:
                # If first subnet and not last, compute the forward pass of the first two layers (Flatten and FC)
                if net.first and not net.last:
                    output = net.layers[0].forward(output)
                    output = net.layers[1].forward(output)

                # If last subnet, compute the forward pass of all the layers
                elif net.last:
                    for layer in net.layers:
                        output = layer.forward(output)

                # Otherwise, compute the forward pass of the first layer only
                else:
                    output = net.layers[0].forward(output)

            if batch == 0:
                outputs = cp.asnumpy(output)
            else:
                outputs = np.concatenate((outputs, cp.asnumpy(output)))

        # Return the predicted output
        if outputs.shape[1] == 1:
            # If binary classification, return the sign of the output
            return np.sign(outputs)
        else:
            return np.argmax(outputs, axis=1)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, args: argparse.Namespace) -> tuple[list[float], list[float]]:
        '''
        Fit the super network on the given dataset. It fits each subnetwork in the super network on the given dataset.

        Parameters
        ----------
        x_train : numpy.ndarray
            Input training dataset.
        y_train : numpy.ndarray
            Target training dataset.
        x_test : numpy.ndarray
            Input test dataset.
        y_test : numpy.ndarray
            Target test dataset.
        args : argparse.Namespace
            Dictionary containing the arguments passed to the program.

        Returns
        -------
        tuple[list[float], list[float]
            Tuple containing the training and test accuracies.
        '''
        test_accs = []
        train_accs = []
        train_acc = 0.0

        # Repeat for the number of epochs
        for e in range(1, args.epochs+1):
            epoch_corr = 0
            num_lines = 0

            # Shuffle the training dataset
            x_train, y_train = shuffle_dataset((x_train, y_train))

            # Loop over the mini batches in the buffer
            start_time = time.time()
            print_count = 0
            num_batches = len(x_train)//args.bs + 1 if len(x_train)%args.bs != 0 else len(x_train)//args.bs
            for mini_batch in range(num_batches):
                batch_corr = 0

                # Compute the starting and ending index of the mini batch
                idx_start_batch = mini_batch * args.bs
                idx_end_batch = idx_start_batch + args.bs
                if idx_start_batch >= len(x_train):
                    break
                elif idx_end_batch > len(x_train):
                    idx_end_batch = len(x_train)

                # Extract the correct mini batch data and target from the training dataset
                x_batch = cp.asarray(x_train[idx_start_batch:idx_end_batch])
                y_batch = cp.asarray(y_train[idx_start_batch:idx_end_batch])

                # Fit the subnetworks on the mini batch
                for net in self.subnets:
                    x_batch = net.fit(x_batch, y_batch, args, train_acc)
                fwd_out = cp.asnumpy(x_batch)
                targets = cp.asnumpy(y_batch)

                # Compute the number of correct predictions in the mini batch
                if self.subnets[-1].layers[-1].output_dim == 1:
                    batch_corr = np.sum(np.sign(targets) == np.sign(fwd_out))
                else:
                    batch_corr = np.sum(np.argmax(targets, axis=1) == np.argmax(fwd_out, axis=1))
                epoch_corr += batch_corr.item()

                # Print the training progress
                if args.log:
                    print_count += 1
                    if print_count == PRINT_INTERVAL:
                        num_lines = self.print_progress(e, args.epochs, mini_batch+1, num_batches, epoch_corr/len(x_train)*100, 0, start_time, num_lines)
                        print_count = 0

            # Compute the test accuracy
            test_acc = self.test(x_test, y_test)
            train_acc = epoch_corr/len(x_train) * 100

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            # Print the training progress
            if args.log:
                self.print_progress(e, args.epochs, mini_batch+1, num_batches, train_acc, test_acc, start_time, num_lines)

        return train_accs, test_accs

    def print_progress(self, epoch: int, epochs: int, batch: int, batches: int, train_acc: float, test_acc: float, start_time: float, num_lines: int = 0) -> int:
        '''
        Print the training progress.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        epochs : int
            Total number of epochs.
        batch : int
            Current batch.
        batches : int
            Total number of batches.
        train_acc : float
            Training accuracy.
        test_acc : float
            Test accuracy.
        start_time : float
            Starting time of the training.
        num_lines : int (default = 0)
            Number of lines to clear from previous prints.
            
        Returns
        -------
        int
            Number of lines printed.
        '''
        # Get the width of the terminal
        width = os.get_terminal_size().columns
        
        completed_bars = int(batch/batches*BARS)
        percentage_completed = int(batch/batches*100)
        epoch_format = f"{{:{len(str(epochs))}d}}"
        batch_format = f"{{:{len(str(batches))}d}}"
        
        # Create the string to print
        string = f"{epoch_format.format(epoch)}/{epochs} [{''.join(['=' for _ in range(completed_bars)])}>{''.join([' ' for _ in range(BARS-completed_bars)])}]{'{:3d}'.format(percentage_completed)}% ({batch_format.format(batch)}/{batches}) - Train acc: {'{:5.2f}'.format(train_acc)}% - Test acc: {'{:5.2f}'.format(test_acc)}% - Total time: {'{:5.2f}'.format(time.time()-start_time)}s \t"

        # Wrap the string to fit within the terminal width
        wrapped_string = textwrap.fill(string, width=width)

        # Clear the previous lines
        for _ in range(num_lines):
            print("\033[A\033[K", end="")
        
        # Save the number of lines to clear in the next iteration
        num_lines = len(wrapped_string.splitlines())
        
        # Print the wrapped string
        print(wrapped_string)
        
        return num_lines

    def __repr__(self) -> str:
        '''
        Return the string representation of the super network.
        '''
        nets = ""
        for idx, net in enumerate(self.subnets):
            nets += f"SubNet {idx}: {net}\n"
        return nets