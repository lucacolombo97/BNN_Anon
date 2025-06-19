import numpy as np
from scipy.special import erfinv

class Thermometer:
    def __init__(self, num_bits=1, feature_wise=True):
        assert num_bits > 0
        assert isinstance(feature_wise, bool)

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        min_value = np.min(x, axis=0) if self.feature_wise else np.min(x)
        max_value = np.max(x, axis=0) if self.feature_wise else np.max(x)
        
        thresholds = (
            min_value[..., None] +
            np.arange(1, self.num_bits + 1)[None, ...] * 
            ((max_value - min_value) / (self.num_bits + 1))[..., None]
        )
        return thresholds

    def fit(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        self.thresholds = self.get_thresholds(x)
        return self

    def binarize(self, x):
        if self.thresholds is None:
            raise ValueError('Need to fit before calling binarize.')
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = x[..., None]
        
        # Binarize the data with +1 and -1 values
        return np.where(x < self.thresholds, -1, 1)
        
class GaussianThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        std_skews = np.linspace(1 / (self.num_bits + 1), self.num_bits / (self.num_bits + 1), self.num_bits)
        std_skews = np.array([np.sqrt(2) * erfinv(2 * p - 1) for p in std_skews])  # Use erfinv from scipy.special
        mean = np.mean(x, axis=0) if self.feature_wise else np.mean(x)
        std = np.std(x, axis=0) if self.feature_wise else np.std(x)
        thresholds = np.stack([std_skew * std + mean for std_skew in std_skews], axis=-1)
        return thresholds

class DistributiveThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        if self.feature_wise:
            data = np.sort(x, axis=0)
            indices = (np.arange(1, self.num_bits + 1) / (self.num_bits + 1) * (x.shape[0] - 1)).astype(int)
            thresholds = data[indices, ...]
        else:
            data = np.sort(x.flatten())
            indices = (np.arange(1, self.num_bits + 1) / (self.num_bits + 1) * (data.size - 1)).astype(int)
            thresholds = data[indices]
        return np.moveaxis(thresholds, 0, -1)