"""


Plot for illustrating the training data



"""

import os

import matplotlib.pyplot as plt

from src.config import dataset_config
from src.data.data_init import get_radial_data

from plots.util import get_dpi, get_figure_folder


def get_illustration_synthetic_train_data(num_classes: int) -> None:
    """ Get a plot illustrating the training data """

    dataset_name_str = 'radial_classification' 
    num_points = 20000     
    random_seed = 0
    equal_weighting_classes = False

    data_config = dataset_config.DatasetConfig(
        random_seed = random_seed, 
        dataset_name = dataset_name_str,
        num_points = num_points,
        num_classes = num_classes,
        equal_weighting_classes = equal_weighting_classes,
        min_length = 0,
        dist_scale = 0,
        vert_dist_scale = 0 
    )

    class_train_dataset, _ = get_radial_data(data_config)

    data = class_train_dataset.data
    classes = class_train_dataset.targets

    file_name = f'train_data_illustration_cls{num_classes}'
    file_path = os.path.join(get_figure_folder(), file_name)
    dpi = get_dpi()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[:, 0], data[:, 1], s=5, c=classes, cmap='Paired')
    fig.tight_layout()
    plt.savefig(file_path, dpi = dpi)
    #fig.show()
    plt.close()
