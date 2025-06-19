import os
import uuid
import math
import json
import pytz
import random
import datetime
import argparse
import cupy as cp
import numpy as np

from supernet import SuperNetwork
from dataset.custom_dataset import CustomDataset
from dataset.utils import prepare_dataset, prepare_binary_dataset, train_test_dataset

timezone = pytz.timezone('Europe/Rome')

def read_args() -> argparse.Namespace:
    '''
    Read the arguments from the command line.
    
    Returns
    -------
    argparse.Namespace
        The arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-layer", default="our", type=str, help="Layer algorithm: our or baldassi")
    
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g. fmnist, cifar10tl, cifar100tl, imagenettetl, uci_`name`)")
    parser.add_argument("--binarize-dataset", default=True, type=bool, action=argparse.BooleanOptionalAction, help="Binarize the dataset using the median value")
    parser.add_argument("--thermometer-bits", default=0, type=int, help="Number of bits for thermometer encoding (for tabular dataset)")

    parser.add_argument("--test-dim", default=10000, type=float, help="Dimension of the test set")

    parser.add_argument("--layers", type=str, help="Hidden layers of the network separated by '_' (e.g. '100_50_25'). For Baldassi, use '-' to separate the single layer and the grouping layer")
    parser.add_argument("--weight-clip", default=None, type=int, help="Clip the weights to the given value (None for no clipping)")
    parser.add_argument("--freeze-first", default=False, type=bool, action=argparse.BooleanOptionalAction, help="Freeze the first layer")
    parser.add_argument("--freeze-last", default=True, type=bool, action=argparse.BooleanOptionalAction, help="Freeze the last layer")

    parser.add_argument("--group-size", default=1, type=int, help="The group size in which the perceptrons are divided during the update")

    parser.add_argument("--bs", type=int, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, help="Epochs")
    
    parser.add_argument("--prob-reinforcement", default=0.0, type=float, help="Random reinforcement probability")
    parser.add_argument("--rob", default=0.25, type=float, help="Robustness parameter")
    
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n-runs", default=1, type=int, help="Number of runs")
    parser.add_argument("--device", default=0, type=int, help="Device ID")
    parser.add_argument("--log", default=True, type=bool, action=argparse.BooleanOptionalAction, help="Print logs in the console")
    
    return parser.parse_args()

def worker(args: argparse.Namespace, run: int, train: tuple[np.ndarray, np.ndarray]) -> dict:
    '''
    Train the network.
    
    Parameters
    ----------
    args : argparse.Namespace
        The arguments of the program.
    run : int
        The run number.
    train : tuple[np.ndarray, np.ndarray]
        The training dataset (x, y).
        
    Returns
    -------
    dict
        The accuracy results (train, test).
    '''
    if args.log:
        print()
        print(f"RUN: {run+1}/{args.n_runs}")
    
    # Set the random seed
    seed = args.seed + run
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

    # Prepare the dataset
    if args.binarize_dataset:
        train_set = prepare_binary_dataset(train, args.thermometer_bits)
    else:
        train_set = prepare_dataset(train)

    # Split the training set into training and validation sets
    (x_train, y_train), (x_test, y_test) = train_test_dataset(train_set, args.test_dim)

    # Count the number of samples in each class
    _, counts_train = np.unique(np.argmax(y_train, axis=1), return_counts=True) if y_train.shape[1] > 1 else np.unique(y_train, return_counts=True)
    _, counts_val = np.unique(y_test, return_counts=True) if y_test.ndim > 1 else np.unique(y_test, return_counts=True)
    _, counts_test = np.unique(y_test, return_counts=True)

    if args.log:
        # Print the number of samples in each class and the percentage of the total
        print(f"Train samples: [{'%, '.join(format(x, '.2f') for x in (counts_train/len(y_train)*100).tolist())}%]")
        print(f"Validation samples: [{'%, '.join(format(x, '.2f') for x in (counts_val/len(y_test)*100).tolist())}%]")
        print(f"Test samples: [{'%, '.join(format(x, '.2f') for x in (counts_test/len(y_test)*100).tolist())}%]")
        print()

    # Net structure
    input_dim = x_train[0,...].size
    output_dim = len(y_train[0]) if y_train.shape[1] > 1 else 1

    # Adjust the layers in case of UCI dataset with Baldassi algorithm
    if args.dataset.split("_")[0] == "uci" and args.algo_layer == "baldassi":
        F = 15
        ratio = 5
        
        # Coefficients of the quadratic equation
        a = ratio
        b = ratio * input_dim
        c = -(input_dim * F + F**2)

        # Solving for O using the quadratic formula
        discriminant = b**2 - 4 * a * c

        O1 = (-b + math.sqrt(discriminant)) / (2 * a)
        O2 = (-b - math.sqrt(discriminant)) / (2 * a)
        
        # Selecting only the positive solution and rounding it
        O = round(O1) if O1 > 0 else round(O2)
        
        layers = f"{O*ratio}-{O}"
        args.layers = layers
    
    net = SuperNetwork(args.layers, input_dim, output_dim, args.freeze_first, args.freeze_last, args.group_size, args.weight_clip)
    if args.log:
        print(net)
    
    # Accuracy before training
    train_acc0 = net.test(x_train, y_train)
    test_acc0 = net.test(x_test, y_test)
    if args.log:
        print(f"{0}/{args.epochs} - Train acc: {'{:.2f}'.format(train_acc0)}% - Test acc: {'{:.2f}'.format(test_acc0)}%")

    # Training
    train_accs, test_accs = net.fit(x_train, y_train, x_test, y_test, args)
        
    return {"train": train_accs, "test": test_accs}

if __name__ == "__main__":
    args = read_args()
    
    if args.log:
        # Print the arguments
        print(args)
    
    # Select the device
    cp.cuda.Device(args.device).use()
    
    # Download the dataset
    dataset_params = {}
    if len(args.dataset.split("_")) > 1:
        repo, name = args.dataset.split("_")[0], args.dataset.split("_")[1]
        dataset_params["uci_name"] = name
    else:
        repo = args.dataset
        
    try:
        train = CustomDataset.load_dataset(repo, dataset_params)
    except NotImplementedError:
        raise ValueError(f"Unknown dataset {args.dataset}")
    
    # Launch the training for each run
    train_results = {}
    for run in range(args.n_runs):
        # Launch the training
        train_results[f"run{run+1}"] = worker(args, run, train)