"""


Get the d_prob distances between trained models 


"""


import os

import itertools

import torch

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelVariationsConfig, ModelConfig
from src.config.train_config import TrainConfig
from src.data.data_init import initialize_dataset

from src.dissimilarity_measures.distribution_distance import d_prob, get_mean_KL_divergence
from src.dissimilarity_measures.representation_measures import get_m_SVD_for_y_pivot_and_lo, get_mSVD_chosen_input_set

from src.file_handling.save_load_json import load_json, save_as_json

from src.file_handling.save_load_model import load_trained_model



def get_d_prob_between_models(
        date_str: str, layer_size: int, num_classes: int,
        extra_suff: str, weight: float, device: str, dist_type: str,
        sum_or_max: str, num_samples: int, use_train_for_distances: bool,
        model_var_config_path: str, dataset_config_path: str, train_config_path: str) -> None:
    """
        Get the distance between model distributions using dist_type.
        dist_type: can be either 'max' or 'mean' 
    """
    size_suff = f'_{layer_size}'
    weight_suff = f'_w{str(weight).replace(".", "_")}' 
    data_suff = '_train' if use_train_for_distances else '_test'  
    
    if model_var_config_path == '':
        model_var_config_path = f'configs/model_variations_config_{num_classes}_classes{extra_suff}.json'

    model_json_dict = load_json(model_var_config_path)
    model_var_config = ModelVariationsConfig(**model_json_dict)
    model_type = model_var_config.model_type

    all_seeds = model_var_config.random_seeds

    if dataset_config_path == '':
        dataset_config_path = f'configs/radial_classification_20000_0_cls{num_classes}.json'
    
    dataset_json_dict = load_json(dataset_config_path)
    dataset_config = DatasetConfig(**dataset_json_dict)

    batch_size = 16
    train_dataloader, test_dataloader = initialize_dataset(
        dataset_config, batch_size=batch_size, shuffle_train=False)
    if use_train_for_distances:
        use_dataloader = train_dataloader
    else:
        use_dataloader = test_dataloader

    rep_dim = model_var_config.rep_dim

    if train_config_path == '':
        train_config_path = 'configs/radial_classification_0_128_ADAM_0_0001.json'
    
    train_json_dict = load_json(train_config_path)
    train_config = TrainConfig(**train_json_dict)

    steps = train_config.train_steps
    step_suff = f'_{steps}'

    checkpoint_folder = 'checkpoints'    

    # We go though all choices possible diversity sets for our choice of y pivot 
    # and choice of leave out label 
    possible_diversity_targets_g = range(num_classes-2)
    possible_diversity_combinations_g = list(
        itertools.combinations(possible_diversity_targets_g, model_var_config.rep_dim))
    
    for current_fix_g_option in model_var_config.fix_length_gs:
        for current_fix_f_option in model_var_config.fix_length_fs:
            distance_dict = {
            'model_seeds': [],
            'KL_div': [],
            'distance': [],
            't_1': [],
            't_2': [],
            'm_SVDs_f': [],
            'm_SVDs_g': [],
            'input_idxs': [],
            'y_pivot_idx': [],
            'lo_idx': []
            }
            for k, seed1 in enumerate(all_seeds[:-1]):
                model_config1 = ModelConfig(
                        seed1, 
                        model_var_config.model_type, 
                        model_var_config.target_type, 
                        model_var_config.nonlinearity,
                        num_classes,
                        layer_size,
                        model_var_config.rep_dim,
                        current_fix_g_option,
                        current_fix_f_option
                    )
                model1 = load_trained_model(
                        model_config1, train_config, dataset_config, 
                        date_str, model_config1.random_seed, checkpoint_folder,
                        device=device)
                model1.eval()

                                
                f1_rep_list = []

                with torch.no_grad():
                    for current_data in use_dataloader:
                        imgs, _ = current_data
                        f1_rep_list.append(model1.get_f_reps(imgs))
                
                f1_reps = torch.concatenate(f1_rep_list, dim=0)

                for seed2 in all_seeds[(k+1):]:            
                    model_config2 = ModelConfig(
                        seed2, 
                        model_var_config.model_type, 
                        model_var_config.target_type, 
                        model_var_config.nonlinearity,
                        num_classes,
                        layer_size,
                        model_var_config.rep_dim,
                        current_fix_g_option,
                        current_fix_f_option
                    )

                    model2 = load_trained_model(
                        model_config2, train_config, dataset_config, 
                        date_str, model_config2.random_seed, checkpoint_folder,
                        device=device)
                    model2.eval()

                    possible_targets_oh = model1.possible_targets_oh

                    f2_rep_list = []

                    with torch.no_grad():
                        for current_data in use_dataloader:
                            imgs, _ = current_data
                            f2_rep_list.append(model2.get_f_reps(imgs))
                    
                    f2_reps = torch.concatenate(f2_rep_list, dim=0)

                    rep_dim = model_config1.rep_dim

                    with torch.no_grad():
                        all_g_1_ys = model1.get_g_reps(possible_targets_oh)
                        all_g_2_ys = model2.get_g_reps(possible_targets_oh)

                    g_indexes = torch.arange(all_g_1_ys.shape[0])

                    
                    g1_reps = all_g_1_ys
                    g2_reps = all_g_2_ys

                    # Get KL-divergence
                    current_kl_div = get_mean_KL_divergence(
                        f1_reps, g1_reps, f2_reps, g2_reps, device=device
                    )
                    print(f'{seed1}vs{seed2} mean KL: {current_kl_div}')
                    
                    # Getting distance 
                    distance, t_1, t_2, div_input_idxs, y_pivot_idx, lo_idx = d_prob(
                        f1_reps, g1_reps, f2_reps, g2_reps, num_classes,
                        weight, num_samples=num_samples,
                        device=device, dist_type=dist_type, sum_or_max=sum_or_max)
                    print(f'{seed1}vs{seed2} d_prob: {distance}')
                    
                    # Getting mSVDs
                    mean_svds_f = get_m_SVD_for_y_pivot_and_lo(
                    rep_dim=rep_dim, y_pivot_idx=y_pivot_idx.numpy(), 
                    lo_idx=lo_idx.numpy(),
                    div_rep_indexes=g_indexes.numpy(), 
                    model1_div_reps=all_g_1_ys, model2_div_reps=all_g_2_ys,
                    model1_f_reps=f1_reps, model2_f_reps=f2_reps, 
                    possible_diversity_combinations=possible_diversity_combinations_g
                    )
                    mean_svd_cov_g = get_mSVD_chosen_input_set(
                        div_input_idxs, f1_reps, f2_reps, g1_reps, g2_reps, 
                        num_classes=num_classes, rep_dim=rep_dim
                    )
                    mean_svds_g = [mean_svd_cov_g]
                    print(f'{seed1}vs{seed2} mSVD gs: {mean_svds_g}')

                    distance_dict['model_seeds'].append(f'{seed1}vs{seed2}')
                    distance_dict['KL_div'].append(float(current_kl_div))
                    distance_dict['distance'].append(float(distance))
                    distance_dict['t_1'].append(float(t_1))
                    distance_dict['t_2'].append(float(t_2))
                    distance_dict['m_SVDs_f'].append(mean_svds_f)
                    distance_dict['m_SVDs_g'].append(mean_svds_g)
                    distance_dict['input_idxs'].append(div_input_idxs)
                    distance_dict['y_pivot_idx'].append(y_pivot_idx)
                    distance_dict['lo_idx'].append(lo_idx)

            if current_fix_g_option == 0:
                fix_g_suff = ''
            else:
                fix_g_suff = f'_g_{current_fix_g_option}'
            
            if current_fix_f_option == 0:
                fix_f_suff = ''
            else:
                fix_f_suff = f'_f_{current_fix_f_option}'
            
            if sum_or_max == 'sum':
                sm_suff = ''
            else:
                sm_suff = sum_or_max
            
            if model_type == 'SmallMLP':
                file_name = f'{date_str}_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}{fix_g_suff}{fix_f_suff}{size_suff}_cls{num_classes}{extra_suff}.json'
            elif model_type in ('SmallCIFAR10', 'MedCIFAR10', 'ResNetCIFAR10'):
                file_name = f'{date_str}_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}{fix_g_suff}{fix_f_suff}{size_suff}_fd{model_var_config.rep_dim}{step_suff}.json' 
            else:
                raise ValueError(f'Did not recognize model type: {model_type}') 

            result_folder = 'results'
            file_path = os.path.join(result_folder, file_name)

            save_as_json(distance_dict, file_path)


def get_d_prob_between_different_size_synth_models(
        date_str: str, layer_size1: int, layer_size2: int, num_classes: int,
        extra_suff: str, weight: float, device: str, dist_type: str,
        sum_or_max: str, num_samples: int, use_train_for_distances: bool) -> None:
    """
        Get the distance between model distributions using dist_type.
        dist_type: can be either 'max' or 'mean' 
    """
    vs_size_suff = f'_{layer_size1}vs{layer_size2}'
    weight_suff = f'_w{str(weight).replace(".", "_")}' 
    data_suff = '_train' if use_train_for_distances else '_test'  
    
    model_var_config_path = f'configs/model_variations_config_{num_classes}_classes{extra_suff}.json'

    model_json_dict = load_json(model_var_config_path)
    model_var_config = ModelVariationsConfig(**model_json_dict)
    model_type = model_var_config.model_type

    all_seeds = model_var_config.random_seeds

    dataset_config_path = f'configs/radial_classification_20000_0_cls{num_classes}.json'
    
    dataset_json_dict = load_json(dataset_config_path)
    dataset_config = DatasetConfig(**dataset_json_dict)

    batch_size = 16
    train_dataloader, test_dataloader = initialize_dataset(
        dataset_config, batch_size=batch_size, shuffle_train=False)
    if use_train_for_distances:
        use_dataloader = train_dataloader
    else:
        use_dataloader = test_dataloader

    rep_dim = model_var_config.rep_dim

    train_config_path = 'configs/radial_classification_0_128_ADAM_0_0001.json'
    
    train_json_dict = load_json(train_config_path)
    train_config = TrainConfig(**train_json_dict)

    checkpoint_folder = 'checkpoints'    

    fix_gs_options = [0, 1]
    fix_fs_options = [0, 1]

    # We go though all choices possible diversity sets for our choice of y pivot 
    # and choice of leave out label 
    possible_diversity_targets_g = range(num_classes-2)
    possible_diversity_combinations_g = list(
        itertools.combinations(possible_diversity_targets_g, model_var_config.rep_dim))
    
    for current_fix_g_option in fix_gs_options:
        for current_fix_f_option in fix_fs_options:
            distance_dict = {
            'model_seeds': [],
            'KL_div': [],
            'distance': [],
            't_1': [],
            't_2': [],
            'm_SVDs_f': [],
            'm_SVDs_g': [],
            'input_idxs': [],
            'y_pivot_idx': [],
            'lo_idx': []
            }
            for seed1 in all_seeds: 
                model_config1 = ModelConfig(
                        seed1, 
                        model_var_config.model_type, 
                        model_var_config.target_type, 
                        model_var_config.nonlinearity,
                        num_classes,
                        layer_size1,
                        model_var_config.rep_dim,
                        model_var_config.fix_length_gs[current_fix_g_option],
                        model_var_config.fix_length_fs[current_fix_f_option]
                    )
                model1 = load_trained_model(
                        model_config1, train_config, dataset_config, 
                        date_str, model_config1.random_seed, checkpoint_folder,
                        device=device)
                model1.eval()

                                
                f1_rep_list = []

                with torch.no_grad():
                    for current_data in use_dataloader:
                        imgs, _ = current_data
                        f1_rep_list.append(model1.get_f_reps(imgs))
                
                f1_reps = torch.concatenate(f1_rep_list, dim=0)

                for seed2 in all_seeds:            
                    model_config2 = ModelConfig(
                        seed2, 
                        model_var_config.model_type, 
                        model_var_config.target_type, 
                        model_var_config.nonlinearity,
                        num_classes,
                        layer_size2,
                        model_var_config.rep_dim,
                        model_var_config.fix_length_gs[current_fix_g_option],
                        model_var_config.fix_length_fs[current_fix_f_option]
                    )

                    model2 = load_trained_model(
                        model_config2, train_config, dataset_config, 
                        date_str, model_config2.random_seed, checkpoint_folder,
                        device=device)
                    model2.eval()

                    possible_targets_oh = model1.possible_targets_oh

                    f2_rep_list = []

                    with torch.no_grad():
                        for current_data in use_dataloader:
                            imgs, _ = current_data
                            f2_rep_list.append(model2.get_f_reps(imgs))
                    
                    f2_reps = torch.concatenate(f2_rep_list, dim=0)

                    rep_dim = model_config1.rep_dim

                    with torch.no_grad():
                        all_g_1_ys = model1.get_g_reps(possible_targets_oh)
                        all_g_2_ys = model2.get_g_reps(possible_targets_oh)

                    g_indexes = torch.arange(all_g_1_ys.shape[0])

                    
                    g1_reps = all_g_1_ys
                    g2_reps = all_g_2_ys

                    # Get KL-divergence
                    current_kl_div = get_mean_KL_divergence(
                        f1_reps, g1_reps, f2_reps, g2_reps, device=device
                    )
                    print(f'{seed1}vs{seed2} mean KL: {current_kl_div}')
                    
                    # Getting distance 
                    distance, t_1, t_2, div_input_idxs, y_pivot_idx, lo_idx = d_prob(
                        f1_reps, g1_reps, f2_reps, g2_reps, num_classes,
                        weight, num_samples=num_samples,
                        device=device, dist_type=dist_type, sum_or_max=sum_or_max)
                    print(f'{seed1}vs{seed2} d_prob: {distance}')
                    
                    # Getting mSVDs
                    mean_svds_f = get_m_SVD_for_y_pivot_and_lo(
                    rep_dim=rep_dim, y_pivot_idx=y_pivot_idx.numpy(), 
                    lo_idx=lo_idx.numpy(),
                    div_rep_indexes=g_indexes.numpy(), 
                    model1_div_reps=all_g_1_ys, model2_div_reps=all_g_2_ys,
                    model1_f_reps=f1_reps, model2_f_reps=f2_reps, 
                    possible_diversity_combinations=possible_diversity_combinations_g
                    )
                    mean_svd_cov_g = get_mSVD_chosen_input_set(
                        div_input_idxs, f1_reps, f2_reps, g1_reps, g2_reps, 
                        num_classes=num_classes, rep_dim=rep_dim
                    )
                    mean_svds_g = [mean_svd_cov_g]
                    print(f'{seed1}vs{seed2} mSVD gs: {mean_svds_g}')

                    distance_dict['model_seeds'].append(f'{seed1}vs{seed2}')
                    distance_dict['KL_div'].append(float(current_kl_div))
                    distance_dict['distance'].append(float(distance))
                    distance_dict['t_1'].append(float(t_1))
                    distance_dict['t_2'].append(float(t_2))
                    distance_dict['m_SVDs_f'].append(mean_svds_f)
                    distance_dict['m_SVDs_g'].append(mean_svds_g)
                    distance_dict['input_idxs'].append(div_input_idxs)
                    distance_dict['y_pivot_idx'].append(y_pivot_idx)
                    distance_dict['lo_idx'].append(lo_idx)

            if current_fix_g_option == 0:
                fix_g_suff = ''
            else:
                fix_g_suff = f'_g_{model_var_config.fix_length_gs[current_fix_g_option]}'
            
            if current_fix_f_option == 0:
                fix_f_suff = ''
            else:
                fix_f_suff = f'_f_{model_var_config.fix_length_fs[current_fix_f_option]}'
            
            if sum_or_max == 'sum':
                sm_suff = ''
            else:
                sm_suff = sum_or_max
            
            if model_type == 'SmallMLP':
                file_name = f'{date_str}_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}{fix_g_suff}{fix_f_suff}{vs_size_suff}_cls{num_classes}{extra_suff}.json'
            else:
                raise ValueError(f'Model type not implemented: {model_type}') 

            result_folder = 'results'
            file_path = os.path.join(result_folder, file_name)

            save_as_json(distance_dict, file_path)
