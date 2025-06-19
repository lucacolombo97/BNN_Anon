import os
import numpy as np

from dataset.abstract import Dataset

class CIFAR100Features(Dataset):
    '''
    Class representing the extracted features of the CIFAR100 dataset from a CNN.
    '''
    def __init__(self):
        '''
        Class representing the extracted features of the CIFAR100 dataset from a CNN.
        '''
        self.name = 'cifar100tl'

    def get_dataset(self) -> tuple:
        '''
        Load the dataset and return it as a tuple.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to load.
            
        Returns
        -------
        tuple
            Tuple containing the dataset.
        '''
        # Set correct path
        path = os.path.join('.data', self.name)
        
        # Load the dataset from the folder
        features_path = os.path.join(path, 'features.npy')
        labels_path = os.path.join(path, 'labels.npy')

        required_files = [features_path, labels_path]
        missing_files = [f for f in required_files if not os.path.isfile(f)]
        if missing_files:
            raise FileNotFoundError(
                f"Missing files: {', '.join(missing_files)}. "
                "Please run the `python extract_features.py --dataset cifar100` script to generate the required files."
            )

        # Load the dataset from the folder
        data = np.load(features_path)
        labels = np.squeeze(np.load(labels_path))

        return (data, labels)