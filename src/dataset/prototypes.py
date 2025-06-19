import os
import numpy as np

from dataset.abstract import Dataset

class Prototypes(Dataset):
    '''
    Class representing a prototypes dataset.
    '''
    def __init__(self):
        '''
        Class representing a prototypes dataset.
        '''
        self.name = 'prototypes'
        self.n_samples = 12000
        self.n_features = 1000
        self.n_classes = 10
        self.p = 0.44

    def get_dataset(self) -> tuple:
        '''
        Generate the prototypes dataset.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to load.
            
        Returns
        -------
        tuple
            Tuple containing the dataset.
        '''
        
        def _generate_prototypes_binary_data(n_samples, n_features, n_classes, p, random_state=None, max_tries_factor=10):
            """
            Generate a prototypes dataset of binary vectors with a given number of classes.
            
            Parameters
            ----------
            n_samples : int
                Total number of samples to generate.
            n_features : int
                Dimensionality of each sample (number of binary features).
            n_classes : int
                Number of classes to generate.
            p : float
                Probability of flipping each bit from the class prototype.
            random_state : int or None
                Seed for the random number generator for reproducibility (optional).
                
            Returns
            -------
            X : np.ndarray, shape (n_samples, n_features)
                The generated samples with binary entries (0 or 1).
            y : np.ndarray, shape (n_samples,)
                The class labels for each generated sample.
            """
            # Random generator (for reproducibility if random_state is set)
            rng = np.random.default_rng(seed=random_state)

            # Step 1: Create a random prototype for each class (entries in {-1, +1})
            # Shape: (n_classes, n_features)
            prototypes = rng.choice([-1, +1], size=(n_classes, n_features))

            # Distribute samples roughly equally across classes
            samples_per_class = n_samples // n_classes
            remainder = n_samples % n_classes

            # Prepare lists to collect samples (X) and labels (y)
            X_list = []
            y_list = []

            # Keep a set of encountered samples (as tuples) for global de-duplication
            seen_samples = set()

            for class_label in range(n_classes):
                # Number of samples for this class
                n_class_samples = samples_per_class + (1 if class_label < remainder else 0)

                prototype = prototypes[class_label]

                # Generate unique samples for this class
                generated_count = 0
                tries = 0
                max_tries = n_class_samples * max_tries_factor

                while generated_count < n_class_samples:
                    if tries > max_tries:
                        # Safety check: in case we cannot find enough unique samples
                        raise ValueError(
                            "Could not generate enough unique samples. "
                            f"Needed {n_class_samples}, got {generated_count}. "
                            f"Consider increasing max_tries_factor or reducing n_samples."
                        )
                    tries += 1

                    # Step 2: Flip bits with probability p
                    flip_mask = rng.random(n_features) < p
                    # Start from the prototype
                    sample = prototype.copy()
                    # Flip sign where flip_mask is True
                    sample[flip_mask] *= -1

                    # Convert to tuple for set membership checks
                    sample_tuple = tuple(sample.tolist())

                    if sample_tuple not in seen_samples:
                        # Unique sample found; record it
                        seen_samples.add(sample_tuple)
                        X_list.append(sample)
                        y_list.append(class_label)
                        generated_count += 1

            # Convert lists to arrays
            X = np.array(X_list, dtype=int)
            y = np.array(y_list, dtype=int)

            return X, y

        # Generate the dataset
        train_data, train_labels = _generate_prototypes_binary_data(n_samples=self.n_samples, n_features=self.n_features, n_classes=self.n_classes, p=self.p, random_state=42)
        
        os.makedirs(f'.data/{self.name}', exist_ok=True)
        data_path = f'.data/{self.name}/data.npy'
        labels_path = f'.data/{self.name}/labels.npy'

        if not (os.path.exists(data_path) and os.path.exists(labels_path)):
            np.save(data_path, train_data.astype(np.float32))
            np.save(labels_path, train_labels.astype(np.int32))
        
        return (train_data, train_labels)