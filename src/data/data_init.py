"""

Initialize a dataset


"""


import numpy as np

from torch.utils.data import DataLoader

from src.config.dataset_config import DatasetConfig
from src.data.radial_classification import RadialClassification
from src.data.cifar10_data import load_cifar10



def initialize_dataset(
    dataset_config: DatasetConfig, batch_size: int, shuffle_train: bool = True):
    """ Initializing the dataset """

    match dataset_config.dataset_name:
        case 'radial_classification':
            train_dataset = RadialClassification(
                dataset_config.num_points, dataset_config.num_classes, 
                dataset_config.random_seed,
                equal_weighting_of_classes = dataset_config.equal_weighting_classes)
            test_dataset = RadialClassification(
                int(np.floor(dataset_config.num_points/10)), 
                dataset_config.num_classes, 
                dataset_config.random_seed + 1,
                equal_weighting_of_classes = dataset_config.equal_weighting_classes)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        case 'cifar10':
            train_dataloader, test_dataloader, _ = load_cifar10(batch_size, shuffle_train)
        case _:
            raise ValueError(f'Dataset not recognized: {dataset_config.dataset_name}')
    
    return train_dataloader, test_dataloader


def get_radial_data(dataset_config: DatasetConfig):
    """ Get the synthetic radial data """
    train_dataset = RadialClassification(
        dataset_config.num_points, dataset_config.num_classes, 
        dataset_config.random_seed,
        equal_weighting_of_classes = dataset_config.equal_weighting_classes)
    test_dataset = RadialClassification(
        int(np.floor(dataset_config.num_points/10)), 
        dataset_config.num_classes, 
        dataset_config.random_seed + 1,
        equal_weighting_of_classes = dataset_config.equal_weighting_classes)
    
    return train_dataset, test_dataset
