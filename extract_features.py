import os
import copy
import torch
import argparse
import numpy as np
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
def get_dataset(dataset: str) -> tuple[DataLoader, DataLoader]:
    """
    Get the dataset and return the train and test loaders.
    
    Parameters
    ----------
    dataset : str
        The name of the dataset to load. Options are 'cifar10', 'cifar100', or 'imagenette'.
        
    Returns
    -------
    tuple[DataLoader, DataLoader]
        The train and test data loaders.
    """
    if dataset not in ["cifar10", "cifar100", "imagenette"]:
        raise ValueError("Dataset must be one of 'cifar10', 'cifar100', or 'imagenette'.")
    
    # Transform the data into 'tensors' using the 'transforms' module
    transform = transforms.Compose([
        # Resize to 224x224 (height x width) to match AlexNet input size
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download training dataset
    download = True if not os.path.exists(f'./.data/{dataset}') else False
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root='./.data/cifar10', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./.data/cifar10', train=False, transform=transform, download=True)
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root='./.data/cifar100', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100(root='./.data/cifar100', train=False, transform=transform, download=True)
    elif dataset == "imagenette":        
        train_dataset = datasets.Imagenette(root='./.data/imagenette', split="train", transform=transform, download=download)
        test_dataset = datasets.Imagenette(root='./.data/imagenette', split="val", transform=transform, download=download)
        
    # Create the train and test loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=0, shuffle=False)
    
    return train_loader, test_loader
    
def extract_features(train_loader: DataLoader, test_loader: DataLoader, dataset: str) -> None:
    """
    Extract features from the dataloader using a pre-trained AlexNet model and save them to files.
    
    Parameters
    ----------
    train_loader : DataLoader
        The DataLoader for the training dataset.
    test_loader : DataLoader
        The DataLoader for the test dataset.
    dataset : str
        The name of the dataset being used (e.g., 'cifar10', 'cifar100', 'imagenette').
    """
    # Load a pre-trained AlexNet model
    model = models.alexnet(pretrained=True).features
    model = torch.nn.Sequential(*list(model.children()))
    
    # Reduce the dimension of the output for imagenette dataset since it is simpler
    if dataset == "imagenette":
        model[-1].kernel_size = 5
        model[-1].stride = 3
    print(model)
    
    # Set the model to evaluation mode and move it to the appropriate device
    model.eval()
    model = model.to(device)

    # Extract features from the dataset
    train_features = []
    train_labels = []
    total_batches = len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        with torch.no_grad():
            features = model(inputs)
            features_np = features.cpu().numpy()
            labels_np = labels.numpy()
            
            # Append the features and labels to the lists
            train_features.append(features_np)
            train_labels.append(labels_np)

        # Print the progress bar
        print_progress_bar(batch_idx, total_batches, prefix='Progress:', suffix='Complete', length=50)

    train_features_np = copy.deepcopy(np.concatenate(train_features, axis=0))
    train_labels_np = copy.deepcopy(np.concatenate(train_labels, axis=0))

    print(train_features_np.shape)
    
    # If dataset is imagenette, extract also the test features to increase the training set size
    if dataset == "imagenette":
        test_features = []
        test_labels = []
        total_batches = len(test_loader)
        for batch_idx, (inputs, labels) in enumerate(test_loader, 1):
            inputs = inputs.to(device)
            with torch.no_grad():
                features = model(inputs)
                features_np = features.cpu().numpy()
                labels_np = labels.numpy()
                
                # Append the features and labels to the lists
                test_features.append(features_np)
                test_labels.append(labels_np)

            # Print the progress bar
            print_progress_bar(batch_idx, total_batches, prefix='Progress:', suffix='Complete', length=50)

        test_features_np = np.concatenate(test_features, axis=0)
        test_labels_np = np.concatenate(test_labels, axis=0)

        # Concatenate the train and test data
        train_features_np = np.concatenate([train_features_np, test_features_np], axis=0)
        train_labels_np = np.squeeze(np.concatenate([train_labels_np, test_labels_np], axis=0))
    
    # If the directory does not exist, create it
    os.makedirs(f'.data/{dataset}tl', exist_ok=True)

    # Save features and labels to files
    np.save(f'.data/{dataset}tl/features.npy', train_features_np)
    np.save(f'.data/{dataset}tl/labels.npy', train_labels_np)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Print a progress bar to the console.
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total: 
        print()

def read_args() -> argparse.Namespace:
    '''
    Read the arguments from the command line.
    
    Returns
    -------
    argparse.Namespace
        The arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset: cifar10, cifar100, imagenette")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    
    train_loader, test_loader = get_dataset(args.dataset)
    extract_features(train_loader, test_loader, args.dataset)