import os
import gzip
import numpy as np
from urllib.request import urlretrieve

from dataset.abstract import Dataset

class FashionMNIST(Dataset):
    '''
    Class representing the FashionMNIST dataset.
    '''
    def __init__(self):
        '''
        Class representing the FashionMNIST dataset.
        '''
        self.name = "fmnist"
        self.url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
        self.files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']

    def get_dataset(self) -> tuple:
        '''
        Load the dataset from the web and return it as a tuple.
        
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

        # Create path if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Download any missing files
        for file in self.files:
            if file not in os.listdir(path):
                urlretrieve(self.url + file, os.path.join(path, file))
                print("Downloaded %s to %s" % (file, path))

        def _images(path: str) -> np.ndarray:
            '''
            Return images loaded locally.
            
            Parameters
            ----------
            path : str
                Path to the zipped images.
                
            Returns
            -------
            np.ndarray
                Numpy array containing the images.
            '''
            with gzip.open(path) as f:
                # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
                samples = np.frombuffer(f.read(), 'B', offset=16)
            return samples.reshape(-1, 28, 28).astype('int32')

        def _labels(path: str) -> np.ndarray:
            '''
            Return labels loaded locally.
            
            Parameters
            ----------
            path : str
                Path to the zipped labels.
                
            Returns
            -------
            np.ndarray
                Numpy array containing the labels.
            '''
            with gzip.open(path) as f:
                # First 8 bytes are magic_number, n_labels
                labels = np.frombuffer(f.read(), 'B', offset=8)
            return labels

        data = _images(os.path.join(path, self.files[0]))
        labels = _labels(os.path.join(path, self.files[1]))

        return (data, labels)