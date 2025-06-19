from abc import ABC, abstractmethod

class Dataset(ABC):
    '''
    Abstract class representing a dataset.
    '''
    @abstractmethod
    def get_dataset() -> tuple:
        '''
        Download the dataset from the web and return it as a tuple.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to download.
            
        Returns
        -------
        tuple
            Tuple containing the dataset.
        '''
        pass