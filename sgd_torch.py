import os
import torch
import argparse
import numpy as np
from torch.optim import SGD
import pytorch_lightning as pl
from torchvision import transforms
from torchmetrics.functional import accuracy
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, CrossEntropyLoss, functional as F

class LightningClassifier(pl.LightningModule):
    def __init__(self, input_dim, layers, lr=1e-2, momentum=0, num_classes=10):
        """
        Lightning module for MLP training.
        
        Parameters
        ----------
        input_dim : int
            Input dimension of the model.
        layers : str
            String representation of the layers, e.g. "784_200_100".
        lr : float, optional
            Learning rate for the optimizer, by default 1e-2.
        momentum : float, optional
            Momentum for the SGD optimizer, by default 0.
        num_classes : int, optional
            Number of classes for classification, by default 10.
        """
        super().__init__()

        # Define the model
        self.layers = layers.split("_")
        for i, layer in enumerate(self.layers):
            if i == 0:
                setattr(self, f"layer_{i}", Linear(input_dim, int(layer), bias=False))
            else:
                setattr(self, f"layer_{i}", Linear(int(self.layers[i-1]), int(layer), bias=False))
        setattr(self, f"layer_{i+1}", Linear(int(self.layers[-1]), num_classes, bias=False))

        self.lr = lr
        self.momentum = momentum
        self.loss = CrossEntropyLoss()
        self.num_classes = num_classes

        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the data.
            
        Returns
        -------
        torch.Tensor
            Output tensor containing the logits.
        """
        # Flatten the input tensor
        if len(x.size()) > 2:
            batch_size, channels, width, height = x.size()
        else:
            batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Forward pass through the layers
        for i in range(len(self.layers)):
            x = F.relu(getattr(self, f"layer_{i}")(x))
        x = getattr(self, f"layer_{i+1}")(x)
        
        # Return the logits
        return x

    def training_step(self, batch):
        """
        Training step of the model.
        
        Parameters
        ----------
        batch : tuple
            A tuple containing the input data and labels.
            
        Returns
        -------
        torch.Tensor
            Loss value for the training step.
        """
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        
        self.log("metrics/train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        """
        Validation step of the model.
        
        Parameters
        ----------
        batch : tuple
            A tuple containing the input data and labels.
        
        Returns
        -------
        torch.Tensor
            Predictions for the test step.
        """
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        
        self.log("metrics/val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return preds

    def configure_optimizers(self):
        """
        Define the optimizer for the model (SGD).
        
        Returns
        -------
        torch.optim.Optimizer
            Optimizer for the model parameters.
        """
        return SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

    def _get_preds_loss_accuracy(self, batch):
        """
        Get predictions, loss, and accuracy for a batch.
        
        Parameters
        ----------
        batch : tuple
            A tuple containing the input data and labels.
            
        Returns
        -------
        tuple
            A tuple containing predictions, loss, and accuracy.
        """
        # Get the input data and labels from the batch
        x, y = batch
        
        # Forward pass through the model
        logits = self(x)
        
        # Compute predictions, loss, and accuracy
        preds = torch.argmax(logits, dim=1)
        
        # Compute loss and accuracy
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, "multiclass", num_classes=self.num_classes)

        return preds, loss, acc
  
    def on_train_epoch_end(self) -> None:
        """
        Print the training metrics at the end of each epoch.
        """
        print("\n")
        return super().on_train_epoch_end()
  
class FashionMNISTDataModule(pl.LightningDataModule):
    """
    Data module for full-precision FashionMNIST dataset. 
    """
    def setup(self, stage=None):
        # Transforms for images
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
          
        # Prepare FMNIST dataset
        fmnist_full = FashionMNIST(".data/", train=True, download=True, transform=transform)

        # Split into train and validation
        self.train, self.val = random_split(fmnist_full, [50000, 10000])
        
        # Reduce the training set size for matching memory footprint
        self.train, _ = random_split(self.train, [1560, 48440])

        # Count the number of samples for each class and print the percentage
        class_counts = np.zeros(10)
        for _, label in self.train:
            class_counts[label] += 1
        print("Length of the training set:", len(self.train))
        print("Data distribution in the training set:", 100 * class_counts / class_counts.sum())
        
        # Prepare test set
        self.test = FashionMNIST(".data/", train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=100, num_workers=4)

class FashionMNISTDataModuleBin(pl.LightningDataModule):
    """
    Data module for binarized FashionMNIST dataset. This module applies a binarization transformation based on the median value of each image.
    """
    def setup(self, stage=None):
        # Transforms for images with binarization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > x.median()).float() * 2 - 1)  # Binarize using median value
        ])

        # Prepare FMNIST dataset
        fmnist_full = FashionMNIST(".data/", train=True, download=True, transform=transform)

        # Split into train and validation
        self.train, self.val = random_split(fmnist_full, [50000, 10000])

        # Prepare test set
        self.test = FashionMNIST(".data/", train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=100, num_workers=4)

class NumpyDataset(Dataset):
    """
    A generic dataset class for loading numpy arrays.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        # Apply transformation if defined
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class ImagenetteDataModule(pl.LightningDataModule):
    """
    Data module for full-precision Imagenette dataset. 
    """
    def setup(self, stage=None):
        # Check if the expected data files exist
        if not (os.path.exists(".data/imagenettetl/features.npy") and os.path.exists(".data/imagenettetl/labels.npy")):
            raise FileNotFoundError("Expected data files not found. Please run the `python extract_features.py --dataset imagenette` script to generate the required files.")
        
        # Load the data
        features = np.load(".data/imagenettetl/features.npy")
        labels = np.load(".data/imagenettetl/labels.npy")
        
        # Prepare Imagenette dataset
        imagenette_full = NumpyDataset(features, labels)

        # Split into train and validation
        self.train, self.val = random_split(imagenette_full, [10000, 3394])
        
        # Reduce the training set size for matching memory footprint
        self.train, _ = random_split(self.train, [312, 9688])

        # Count the number of samples for each class and print the percentage
        class_counts = np.zeros(10)
        for _, label in self.train:
            class_counts[label] += 1
        print("Length of the training set:", len(self.train))
        print("Data distribution in the training set:", 100 * class_counts / class_counts.sum())
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

class ImagenetteDataModuleBin(pl.LightningDataModule):
    """
    Data module for binarized Imagenette dataset. This module applies a binarization transformation based on the median value of each image.
    """
    def setup(self, stage=None):
        # Check if the expected data files exist
        if not (os.path.exists(".data/imagenettetl/features.npy") and os.path.exists(".data/imagenettetl/labels.npy")):
            raise FileNotFoundError("Expected data files not found. Please run the `python extract_features.py --dataset imagenette` script to generate the required files.")
        
        # Load the data
        features = np.load(".data/imagenettetl/features.npy")
        labels = np.load(".data/imagenettetl/labels.npy")
                
        # Transforms for images with binarization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > x.median()).float() * 2 - 1)  # Binarize using median value
        ])
        
        # Prepare Imagenette dataset
        imagenette_full = NumpyDataset(features, labels, transform=transform)

        # Split into train and validation
        self.train, self.val = random_split(imagenette_full, [10000, 3394])

        # Count the number of samples for each class and print the percentage
        class_counts = np.zeros(10)
        for _, label in self.train:
            class_counts[label] += 1
        print("Length of the training set:", len(self.train))
        print("Data distribution in the training set:", 100 * class_counts / class_counts.sum())
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

class CIFAR10DataModule(pl.LightningDataModule):
    """
    Data module for full-precision CIFAR10 dataset.
    """
    def setup(self, stage=None):
        # Check if the expected data files exist
        if not (os.path.exists(".data/imagenettetl/features.npy") and os.path.exists(".data/imagenettetl/labels.npy")):
            raise FileNotFoundError("Expected data files not found. Please run the `python extract_features.py --dataset cifar10` script to generate the required files.")
        
        # Load the data
        features = np.load(".data/cifar10tl/features.npy")
        labels = np.load(".data/cifar10tl/labels.npy")
        
        # Prepare CIFAR10 dataset
        cifar10_full = NumpyDataset(features, labels)

        # Split into train and validation
        self.train, self.val = random_split(cifar10_full, [40000, 10000])
        
        # Reduce the training set size for matching memory footprint
        self.train, _ = random_split(self.train, [1250, 38750])

        # Count the number of samples for each class and print the percentage
        class_counts = np.zeros(10)
        for _, label in self.train:
            class_counts[label] += 1
        print("Length of the training set:", len(self.train))
        print("Data distribution in the training set:", 100 * class_counts / class_counts.sum())
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)
    
class CIFAR10DataModuleBin(pl.LightningDataModule):
    """ Data module for binarized CIFAR10 dataset. This module applies a binarization transformation based on the median value of each image.
    """
    def setup(self, stage=None):
        # Check if the expected data files exist
        if not (os.path.exists(".data/imagenettetl/features.npy") and os.path.exists(".data/imagenettetl/labels.npy")):
            raise FileNotFoundError("Expected data files not found. Please run the `python extract_features.py --dataset cifar10` script to generate the required files.")
        
        # Load the data
        features = np.load(".data/cifar10tl/features.npy")
        labels = np.load(".data/cifar10tl/labels.npy")

        # Transforms for images with binarization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > x.median()).float() * 2 - 1)  # Binarize using median value
        ])

        # Prepare CIFAR10 dataset
        cifar10_full = NumpyDataset(features, labels, transform=transform)

        # Split into train and validation
        self.train, self.val = random_split(cifar10_full, [40000, 10000])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

class CIFAR100DataModule(pl.LightningDataModule):
    """
    Data module for full-precision CIFAR100 dataset.
    """
    def setup(self, stage=None):
        # Check if the expected data files exist
        if not (os.path.exists(".data/imagenettetl/features.npy") and os.path.exists(".data/imagenettetl/labels.npy")):
            raise FileNotFoundError("Expected data files not found. Please run the `python extract_features.py --dataset cifar100` script to generate the required files.")
        
        # Load the data
        features = np.load(".data/cifar100tl/features.npy")
        labels = np.load(".data/cifar100tl/labels.npy")
        
        # Prepare CIFAR10 dataset
        cifar100_full = NumpyDataset(features, labels)

        # Split into train and validation
        self.train, self.val = random_split(cifar100_full, [40000, 10000])
        
        # Reduce the training set size for matching memory footprint
        self.train, _ = random_split(self.train, [1250, 38750])

        # Count the number of samples for each class and print the percentage
        class_counts = np.zeros(100)
        for _, label in self.train:
            class_counts[label] += 1
        print("Length of the training set:", len(self.train))
        print("Data distribution in the training set:", 100 * class_counts / class_counts.sum())
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)
    
class CIFAR100DataModuleBin(pl.LightningDataModule):
    """ Data module for binarized CIFAR100 dataset. This module applies a binarization transformation based on the median value of each image.
    """
    def setup(self, stage=None):
        # Check if the expected data files exist
        if not (os.path.exists(".data/imagenettetl/features.npy") and os.path.exists(".data/imagenettetl/labels.npy")):
            raise FileNotFoundError("Expected data files not found. Please run the `python extract_features.py --dataset cifar100` script to generate the required files.")
        
        # Load the data
        features = np.load(".data/cifar100tl/features.npy")
        labels = np.load(".data/cifar100tl/labels.npy")

        # Transforms for images with binarization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > x.median()).float() * 2 - 1)  # Binarize using median value
        ])

        # Prepare CIFAR10 dataset
        cifar100_full = NumpyDataset(features, labels, transform=transform)

        # Split into train and validation
        self.train, self.val = random_split(cifar100_full, [40000, 10000])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

class PrototypesDataModule(pl.LightningDataModule):
    """
    Data module for prototypes dataset.
    """
    def setup(self, stage=None):
        # Check if the expected data files exist
        if not (os.path.exists(".data/imagenettetl/features.npy") and os.path.exists(".data/imagenettetl/labels.npy")):
            raise FileNotFoundError("Expected data files not found. Please run the script for the experiments of our solution with --dataset prototypes to generate the required files.")
        
        # Load the data
        data = np.load(".data/prototypes/data.npy")
        labels = np.load(".data/prototypes/labels.npy")
        
        # Prepare Imagenette dataset
        prototypes_full = NumpyDataset(data, labels)

        # Split into train and validation
        self.train, self.val = random_split(prototypes_full, [10000, 2000])

        # Count the number of samples for each class and print the percentage
        class_counts = np.zeros(10)
        for _, label in self.train:
            class_counts[label] += 1
        print("Length of the training set:", len(self.train))
        print("Data distribution in the training set:", 100 * class_counts / class_counts.sum())
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=100, num_workers=4)
    
def read_args() -> argparse.Namespace:
    '''
    Read the arguments from the command line.
    
    Returns
    -------
    argparse.Namespace
        The arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset: fmnist, cifar10tl, bcifar10tl, cifar100tl, bcifar100tl, imagenettetl, bimagenettetl, synthetic")

    parser.add_argument("--layers", type=str, help="Network structure (e.g. 784_200_100)")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--momentum", default=0.0, type=float, help="Momentum for SGD optimizer")

    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n-runs", default=1, type=int, help="Number of runs")
    
    parser.add_argument("--device", default=0, type=int, help="Device ID")
    
    return parser.parse_args()

def worker(args: argparse.Namespace, n_run: int) -> None:
    print()
    print(f"RUN: {n_run+1}/{args.n_runs}")
    
    # Set the random seed    
    seed = args.seed + n_run
    pl.seed_everything(seed)

    # Prepare the dataset
    num_classes = 10
    input_dim = 784
    match args.dataset:
        case "fmnist":
            data_module = FashionMNISTDataModule()
        case "cifar10tl":
            data_module = CIFAR10DataModule()
            input_dim = 9216
        case "cifar100tl":
            data_module = CIFAR100DataModule()
            input_dim = 9216
            num_classes = 100
        case "imagenettetl":
            data_module = ImagenetteDataModule()
            input_dim = 2304
        case "bimagenettetl":
            data_module = ImagenetteDataModuleBin()
            input_dim = 2304
        case "prototypes":
            data_module = PrototypesDataModule()
            input_dim = 1000
        case "bfmnist":
            data_module = FashionMNISTDataModuleBin()
        case "bcifar10tl":
            data_module = CIFAR10DataModuleBin()
            input_dim = 9216
        case "bcifar100tl":
            data_module = CIFAR100DataModuleBin()
            input_dim = 9216
            num_classes = 100
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Prepare the trainer
    trainer = pl.Trainer(max_epochs=50, devices=[args.device])
    
    # Define the model
    model = LightningClassifier(input_dim=input_dim, layers=args.layers, lr=args.lr, momentum=args.momentum, num_classes=num_classes)
    print(model)
    
    # Training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    args = read_args()
    
    # Print the arguments
    print(args)
    
    # Launch the training for each run
    for n_run in range(args.n_runs):
        # Launch the training
        worker(args, n_run)