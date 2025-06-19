import json
import numpy as np
from ucimlrepo import fetch_ucirepo

from dataset.abstract import Dataset
from dataset.binarization import DistributiveThermometer as Thermometer
from dataset.utils import fill_missing_values, encode_categorical

class UCI(Dataset):
    '''
    Class representing the UCI datasets.
    '''
    def __init__(self, name: str):
        '''
        Class representing the UCI datasets.
        '''        
        with open('src/dataset/uci.json', "r") as json_file:
            uci_datasets = json.load(json_file)
        self.id = uci_datasets[name]

    def get_dataset(self) -> tuple:
        '''
        Load the dataset from the UCI repo and return it as a tuple.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to load.
            
        Returns
        -------
        tuple
            Tuple containing the dataset.
        '''        
        # Download the dataset
        dataset = fetch_ucirepo(id=self.id)
                
        # Retrieve the data and labels
        data = dataset.data.features
        labels = dataset.data.targets
        
        # Fill missing values
        data = data.replace('?', np.nan)
        data = fill_missing_values(data)

        # Find categorical, noncategorical, and numerical columns
        categorical_features = [var["name"] for var in dataset.variables.iloc if var["type"].lower() in ["categorical", "binary"] and var["role"].lower() not in ["target", "id"]]
        numerical_features = [var["name"] for var in dataset.variables.iloc if var["type"].lower() in ["integer", "float", "continuous"] and var["role"].lower() not in ["target", "id"]]

        # Separate categorical and numerical columns from the data
        categorical_columns = data[categorical_features].values
        numerical_columns = data[numerical_features].values
        
        # Initialize the arrays
        encoded_categorical_columns = np.array([[]])
        encoded_numerical_columns = np.array([[]])

        # Encode the columns
        if len(categorical_features) > 0:
            encoded_categorical_columns = encode_categorical(categorical_columns)
        if len(numerical_features) > 0:
            encoder = Thermometer(num_bits=32)
            encoder.fit(numerical_columns)
            encoded_numerical_columns = encoder.binarize(numerical_columns)
            encoded_numerical_columns = np.reshape(encoded_numerical_columns, (encoded_numerical_columns.shape[0], -1))
        
        # Concatenate encoded categorical columns and noncategorical columns
        if encoded_categorical_columns.size > 0 and encoded_numerical_columns.size > 0:
            data = np.concatenate([encoded_categorical_columns, encoded_numerical_columns], axis=1)
        else:
            data = encoded_categorical_columns if encoded_categorical_columns.size > 0 else encoded_numerical_columns
        
        # Train set
        train_data = data.astype(int)
        labels, train_labels = np.unique(labels.values, return_inverse=True)
        return (train_data, train_labels)