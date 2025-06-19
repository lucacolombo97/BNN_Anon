import random
import numpy as np

from dataset.binarization import Thermometer

def shuffle_dataset(set: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    '''
    Shuffle the dataset.
    
    Parameters
    ----------
    set : tuple
        Tuple containing the dataset to shuffle. 
        
    Returns
    -------
    tuple
        Tuple containing the shuffled dataset.
    '''
    # Unpack set
    x, y = set
    
    # Shuffle set
    idx_list = list(range(len(x)))
    random.shuffle(idx_list)
    
    # Return shuffled set
    return (x[idx_list, ...], y[idx_list,])

def train_test_dataset(train_set: tuple[np.ndarray, np.ndarray], test_dim: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Split the training set into training and test sets.
    
    Parameters
    ----------
    train_set : tuple
        Tuple containing the training set (x_train, y_train).
    test_dim : int, optional (default=len(x_train))
        Number of test samples.
        
    Returns
    -------
    tuple
        Tuple containing the training and test sets.
    '''
    # Unpack train set
    x_train, y_train = train_set
    num_classes = len(np.unique(y_train))

    # One-hot encoding y train set if not binary classification
    if num_classes > 2:
        y_train = np.eye(np.unique(y_train).size)[y_train].astype(np.int32)
    else:
        # Encode the binary classes as -1 and 1
        y_train = np.where(y_train == 0, -1, 1).astype(np.int32).reshape(-1, 1)
    
    # Split train and test sets
    if test_dim >= 1:
        x_train, x_test = x_train[:-int(test_dim)], x_train[-int(test_dim):]
        y_train, y_test = y_train[:-int(test_dim)], y_train[-int(test_dim):]
    else:
        x_train, x_test = x_train[:-int(test_dim*len(x_train))], x_train[-int(test_dim*len(x_train)):]
        y_train, y_test = y_train[:-int(test_dim*len(y_train))], y_train[-int(test_dim*len(y_train)):]
        
    # Compute argmax for one-hot encoded labels for test set
    if num_classes > 2:
        y_test = np.argmax(y_test, axis=1)

    return (x_train, y_train), (x_test, y_test)

def prepare_binary_dataset(train_set: tuple[np.ndarray, np.ndarray], thermometer_bits: int = 0, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray]:
    '''
    Prepare the binarized dataset for training. If thermometer_bits is set to 0, the median will be used for binarization, otherwise the thermometer encoding will be used.
    
    Parameters
    ----------
    train_set : tuple
        Tuple containing the training set (x_train, y_train).
    thermometer_bits : int, optional (default=0)
        Number of bits to use for the thermometer encoding. If set to 0, the median will be used for binarization.
    shuffle : bool, optional (default=True)
        Shuffle the training set.
        
    Returns
    -------
    tuple
        Tuple containing the binarized training set.
    '''
    # Shuffle train set if required
    x_train, y_train = shuffle_dataset(train_set) if shuffle else train_set

    # Binarize the set
    if thermometer_bits > 0:
        # Create a thermometer encoder
        encoder = Thermometer(num_bits=thermometer_bits)
        
        # Fit the encoder to the training set
        encoder.fit(x_train)
        
        # Binarize the set using the thermometer encoding
        x_train = encoder.binarize(x_train)
    else:
        # Compute the median of the dataset based on its dimensions
        axis = (0, 2, 3) if len(x_train.shape) == 4 else (0, 1, 2) if len(x_train.shape) == 3 else (0,)

        # Compute per-channel median
        median_train = np.median(x_train, axis=axis, keepdims=True)
    
        # Using the median as threshold based on the dimensions of the set
        x_train = np.where(x_train > median_train, 1, -1)
    
    return (x_train, y_train)
    
def prepare_dataset(train_set: tuple[np.ndarray, np.ndarray], shuffle: bool = True) -> tuple[np.ndarray, np.ndarray]:
    '''
    Prepare the dataset for training without binarization.
    
    Parameters
    ----------
    train_set : tuple[np.ndarray, np.ndarray]
        Tuple containing the training set (x_train, y_train).
    shuffle : bool, optional (default=True)
        Shuffle the training set.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the training set.
    '''
    # Shuffle train set if required
    x_train, y_train = shuffle_dataset(train_set) if shuffle else train_set
    
    return (x_train, y_train)

def encode_categorical(array: np.ndarray) -> np.ndarray:
    '''
    One-hot encode a dataset with categorical features using -1 and 1.

    Parameters
    ----------
    array : np.ndarray
        2D Array with categorical data (shape: samples x features).

    Returns
    -------
    np.ndarray
        One-hot encoded array with shape (samples, values) using -1 and 1 as values.
    '''
    # Validate input
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2-dimensional.")

    # Find unique values for each categorical column
    unique_values = [np.unique(array[:, i]) for i in range(array.shape[1])]

    # One-hot encode each column using -1 and 1
    encoded_columns = [
        (np.eye(len(unique_values[i]))[np.searchsorted(unique_values[i], array[:, i])].astype(int) * 2 - 1)
        for i in range(array.shape[1])
    ]

    # Concatenate all encoded columns horizontally
    encoded_array = np.hstack(encoded_columns)

    return encoded_array.astype(int)

def fill_missing_values(array: np.ndarray) -> np.ndarray:
    '''
    Fill missing values in an array.
    
    Parameters
    ----------
    array : np.ndarray
        Array to fill.
        
    Returns
    -------
    np.ndarray
        Array with filled missing values.
    '''    
    # Find columns with missing values
    missing_features = array.columns[array.isna().any()].tolist()
    
    for column in missing_features:
        array[column].fillna(array[column].mode()[0], inplace=True)
    return array