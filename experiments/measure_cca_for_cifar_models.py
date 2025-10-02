"""

Measuring mean canonical correlation for trained CIFAR-10 models


"""

import os

import numpy as np

from sklearn.cross_decomposition import CCA

import torch
from tqdm import tqdm

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelVariationsConfig, ModelConfig
from src.config.train_config import TrainConfig
from src.data.data_init import initialize_dataset

from src.file_handling.save_load_json import load_json, save_as_json

from src.file_handling.save_load_model import load_trained_model



def get_representations_from_models(model1, model2, dataset):
    f1_reps_list = []
    f2_reps_list = []

    with torch.no_grad():
        for current_batch in tqdm(dataset):
            current_inputs, _ = current_batch

            current_f_1_reps = model1.f_net(current_inputs)
            current_f_2_reps = model2.f_net(current_inputs)

            f1_reps_list.append(current_f_1_reps)
            f2_reps_list.append(current_f_2_reps)

    f_1_reps = torch.cat(f1_reps_list)
    f_2_reps = torch.cat(f2_reps_list)

    possible_targets_oh = model1.possible_targets_oh
    
    with torch.no_grad():
        g_1_reps = model1.get_g_reps(possible_targets_oh)
        g_2_reps = model2.get_g_reps(possible_targets_oh)

    return f_1_reps, f_2_reps, g_1_reps, g_2_reps


def get_mcca_scores_cifar_models(
        final_dimension: int, batch_size: int, device: str,
        file_path: str):
    """
    
    Get the mean canonical correlation scores between embeddings and unembeddings
    for models trained on CIFAR-10.

    """

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

    train_dataset, test_dataset = initialize_dataset(dataset_config, batch_size, False)


    checkpoint_folder = 'checkpoints'
    date_str = '2025-04-22'

    if os.path.exists(file_path):
        mcca_dict = load_json(file_path)
    else:
        mcca_dict = {
            'model_seeds': [],
            'train_f_mcca': [],
            'train_g_mcca': [],
            'test_f_mcca': [],
            'test_g_mcca': []
            }

    for i, current_seed1 in enumerate(all_seeds):
        print(f'Seed 1: {current_seed1}')
        model_config1 = ModelConfig(
            current_seed1, 
            model_var_config.model_type, 
            model_var_config.target_type, 
            model_var_config.nonlinearity,
            model_var_config.num_classes,
            model_var_config.num_features[0],
            model_var_config.rep_dim,
            model_var_config.fix_length_gs[0],
            model_var_config.fix_length_fs[0]
        )
        model1 = load_trained_model(
                model_config1, train_config, dataset_config, 
                date_str, model_config1.random_seed, checkpoint_folder, device)
        model1.eval()

        rep_dim = model_config1.rep_dim

        for current_seed2 in tqdm(all_seeds[(i+1):]):
            current_comparison = f'{current_seed1}vs{current_seed2}'
            if current_comparison in mcca_dict['model_seeds']:
                print(f'Comparison already in dict: {current_comparison}')
                continue           
            
            mcca_dict['model_seeds'].append(f'{current_seed1}vs{current_seed2}')

            model_config2 = ModelConfig(
                current_seed2, 
                model_var_config.model_type, 
                model_var_config.target_type, 
                model_var_config.nonlinearity,
                model_var_config.num_classes,
                model_var_config.num_features[0],
                model_var_config.rep_dim,
                model_var_config.fix_length_gs[0],
                model_var_config.fix_length_fs[0]
            )

            model2 = load_trained_model(
                model_config2, train_config, dataset_config, 
                date_str, model_config2.random_seed, checkpoint_folder, device)
            model2.eval()

            reps = get_representations_from_models(model1, model2, train_dataset)
            f_1_reps, f_2_reps, g_1_reps, g_2_reps = reps


            n_components = rep_dim
            cca = CCA(n_components=n_components, max_iter=1000)
            cca.fit(f_1_reps, f_2_reps)
            X_c, Y_c = cca.transform(f_1_reps, f_2_reps)
            corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
            train_f_mcca = np.mean(corrs)

            cca = CCA(n_components=n_components, max_iter=1000)
            cca.fit(g_1_reps, g_2_reps)
            X_c, Y_c = cca.transform(g_1_reps, g_2_reps)
            corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
            train_g_mcca = np.mean(corrs)

            reps = get_representations_from_models(model1, model2, test_dataset)
            f_1_reps, f_2_reps, g_1_reps, g_2_reps = reps
            
            cca = CCA(n_components=n_components, max_iter=1000)
            cca.fit(f_1_reps, f_2_reps)
            X_c, Y_c = cca.transform(f_1_reps, f_2_reps)
            corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
            test_f_mcca = np.mean(corrs)

            cca = CCA(n_components=n_components, max_iter=1000)
            cca.fit(g_1_reps, g_2_reps)
            X_c, Y_c = cca.transform(g_1_reps, g_2_reps)
            corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
            test_g_mcca = np.mean(corrs)

            mcca_dict['train_f_mcca'].append(train_f_mcca)
            mcca_dict['train_g_mcca'].append(train_g_mcca)
            mcca_dict['test_f_mcca'].append(test_f_mcca)
            mcca_dict['test_g_mcca'].append(test_g_mcca)
        
        save_as_json(mcca_dict, file_path)

    return mcca_dict


def get_and_save_mcca_for_cifar_models(final_dimension, device):
    """
    Get and save mean canonical correlations for CIFAR-10 models with the given 
    representational dimension. 
    """
    
    file_name = f'mcca_cifar_fd{final_dimension}.json'

    result_folder = 'results'
    file_path = os.path.join(result_folder, file_name)

    batch_size=32
    mcca_dict = get_mcca_scores_cifar_models(
        final_dimension, batch_size, device, file_path=file_path)

    save_as_json(mcca_dict, file_path)

