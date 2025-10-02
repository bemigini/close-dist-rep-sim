"""

Making a scatterplot of loss vs representational (dis)similarity


"""

import os 
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelVariationsConfig, ModelConfig
from src.config.train_config import TrainConfig

from src.file_handling.save_load_json import load_json, save_as_json
import src.file_handling.naming as name 

from src.training.training_models import get_metrics_folder

from plots.util import get_dpi, get_figure_folder


def save_final_cifar_loss_to_single_file(final_dimension):
    """
    Save the final train and test loss from the CIFAR-10 models in a single file.
    """
    result_folder = 'results'
    single_file_name = f'loss_cifar_fd{final_dimension}.json'
    single_file_path = os.path.join(result_folder, single_file_name)
    
    model_var_config_path = f'configs/model_variations_config_resnetcifar10_128_fd{final_dimension}.json'
    model_json_dict = load_json(model_var_config_path)
    model_var_config = ModelVariationsConfig(**model_json_dict)

    all_seeds = model_var_config.random_seeds

    dataset_config_path = 'configs/cifar10_0_cls10.json'
    dataset_json_dict = load_json(dataset_config_path)
    dataset_config = DatasetConfig(**dataset_json_dict)

    train_config_path = 'configs/cifar10_0_32_ADAM_0_0001_20000steps.json'
    train_json_dict = load_json(train_config_path)
    train_config = TrainConfig(**train_json_dict)

    date_str = '2025-04-22'

    loss_dict = {
            'model_seed': [],
            'train_loss': [],
            'test_loss': []
            }


    for current_seed in all_seeds:
        print(f'Seed: {current_seed}')
        model_config = ModelConfig(
            current_seed, 
            model_var_config.model_type, 
            model_var_config.target_type, 
            model_var_config.nonlinearity,
            model_var_config.num_classes,
            model_var_config.num_features[0],
            model_var_config.rep_dim,
            model_var_config.fix_length_gs[0],
            model_var_config.fix_length_fs[0]
        )
    
    
        trained_model_file_name = name.get_trained_model_name(
                train_config.dataset_name,
                model_config.num_classes, current_seed, 
                model_config.num_features, 
                model_config.rep_dim, dataset_config.num_points, 
                train_config.learning_rate, train_config.train_steps, 
                model_config.model_type,
                fix_fs=model_config.fix_length_fs, fix_gs=model_config.fix_length_gs)
        trained_model_file_name = date_str + trained_model_file_name
        
        metrics_file_name = trained_model_file_name + '_metrics.json'
        metrics_folder = get_metrics_folder()
        metrics_path = os.path.join(metrics_folder, metrics_file_name)

        current_metrics = load_json(metrics_path)
        current_train_loss = current_metrics["train_loss"][-1]
        current_test_loss = current_metrics["test_loss"][-1]

        loss_dict['model_seed'].append(current_seed)
        loss_dict['train_loss'].append(current_train_loss)
        loss_dict['test_loss'].append(current_test_loss)

    save_as_json(loss_dict, single_file_path)


def plot_cifar_loss_diff_vs_mcca(final_dimensions:List[int]):
    """
    Make scatter plot of test loss difference vs m_CCA
    """
    result_folder = 'results'

    figure_folder = get_figure_folder()
    if figure_folder not in os.listdir():
        os.mkdir(figure_folder)
    dpi = get_dpi()
    colour_to_use = '#002347'
    fontsize = 18

    loss_diff_per_dim = []
    mcca_per_dim = []

    for final_dimension in final_dimensions:
        single_file_name = f'loss_cifar_fd{final_dimension}.json'
        single_file_path = os.path.join(result_folder, single_file_name)

        loss_dict = load_json(single_file_path)
        loss_df = pd.DataFrame.from_dict(loss_dict)

        mcca_file_name = f'mcca_cifar_fd{final_dimension}.json'
        mcca_file_path = os.path.join(result_folder, mcca_file_name)

        mcca_dict = load_json(mcca_file_path)
        mcca_df = pd.DataFrame.from_dict(mcca_dict)

        test_loss_diffs = []
        test_f_mccas = []

        for seed_vs in mcca_df['model_seeds'].values:
            seeds = [int(i) for i in seed_vs.split('vs')]
            test_losses = loss_df[loss_df['model_seed'].isin(seeds)]['test_loss'].values
            current_test_loss_diff = np.abs(test_losses[0] - test_losses[1])

            current_test_f_mcca = mcca_df[mcca_df['model_seeds'] == seed_vs]['test_f_mcca'].values[0]

            test_loss_diffs.append(current_test_loss_diff)
            test_f_mccas.append(current_test_f_mcca)

        loss_diff_per_dim.append(test_loss_diffs)
        mcca_per_dim.append(test_f_mccas)


    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(13, 4.6))

    for i, current_mccas in enumerate(mcca_per_dim):
        axs[i].scatter(current_mccas, loss_diff_per_dim[i], s = 4, c=colour_to_use)
        axs[i].set_title(f'{final_dimensions[i]}', fontsize = fontsize)
        axs[i].set_xlabel('embedding mCCA', fontsize = fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize-2)
        
    axs[0].set_ylabel('test loss difference', fontsize = fontsize)   
    

    fig.tight_layout()
    #fig.show()

    figure_name = 'test_loss_diff_vs_embedding_mcca.png'
    figure_path = os.path.join(figure_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


def plot_all_cifar_loss_diff_vs_mcca():
    """
    Make all scatter plots of test loss difference vs m_CCA
    """
    result_folder = 'results'
    final_dimensions = [2, 3, 5]

    for final_dimension in final_dimensions:
        single_file_name = f'loss_cifar_fd{final_dimension}.json'
        single_file_path = os.path.join(result_folder, single_file_name)
        if not os.path.isfile(single_file_path):
            save_final_cifar_loss_to_single_file(final_dimension)

    plot_cifar_loss_diff_vs_mcca(final_dimensions) 
    