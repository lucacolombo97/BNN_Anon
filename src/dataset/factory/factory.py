from dataset.abstract import Dataset

from dataset.uci import UCI
from dataset.fmnist import FashionMNIST
from dataset.prototypes import Prototypes
from dataset.cifar10_tl import CIFAR10Features
from dataset.cifar100_tl import CIFAR100Features
from dataset.imagenette_tl import ImagenetteFeatures

class DatasetFactory:
    '''
    Factory class to create the correct dataset class.
    '''
    @staticmethod
    def get_custom_dataset_handler(dataset: str, dataset_params: dict = None) -> Dataset:
        '''
        Get the correct dataset class for the given dataset name.
        
        Parameters
        ----------
        dataset : str
            Name of the dataset.
        dataset_params : dict, optional (default=None)
            Dictionary containing the dataset parameters.
        '''
        match dataset:
            case "prototypes":
                return Prototypes()
            case "fmnist":
                return FashionMNIST()
            case "cifar10tl":
                return CIFAR10Features()
            case "cifar100tl":
                return CIFAR100Features()
            case "imagenettetl":
                return ImagenetteFeatures()
            case "uci":
                return UCI(dataset_params["uci_name"])
            case _:
                raise NotImplementedError