from dataset.abstract import Dataset
from dataset.factory import DatasetFactory

class CustomDataset:
    '''
    Custom dataset for the different datasets. This class is a wrapper around the dataset classes.
    '''
    @staticmethod
    def load_dataset(dataset: str, dataset_params: dict = None) -> tuple:
        '''
        Load the dataset and return it as a tuple.
        
        Parameters
        ----------
        dataset : str
            Name of the dataset.
        dataset_params : dict, optional (default=None)
            Dictionary containing the dataset parameters.
            
        Returns
        -------
        tuple
            Tuple containing the dataset.           
        '''
        dateset_handler : Dataset = DatasetFactory.get_custom_dataset_handler(dataset, dataset_params)
        return dateset_handler.get_dataset()