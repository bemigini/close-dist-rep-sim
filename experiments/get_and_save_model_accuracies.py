""" 

Get and save model accuracies for trained models


"""



import os

import torch

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelVariationsConfig, ModelConfig
from src.config.train_config import TrainConfig
from src.data.data_init import initialize_dataset

from src.file_handling.save_load_json import load_json, save_as_json

from src.file_handling.save_load_model import load_trained_model

from src.util import convert_one_hot_to_ints




def get_and_save_accuracies(
        date_str: str, layer_size: int, num_classes: int,
        extra_suff: str, device: str, use_train_for_acc: bool,
        model_var_config_path: str, dataset_config_path: str, train_config_path: str) -> None:
    """
        Get the distance between model distributions using dist_type.
        dist_type: can be either 'max' or 'mean' 
    """
    size_suff = f'_{layer_size}'
    data_suff = '_train' if use_train_for_acc else '_test'  
    
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
    if use_train_for_acc:
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
    
    accuracy_dict = {
        'model_seed': [],
        'fix_g': [],
        'fix_f': [],
        'num_classes': [],
        'fd': [],
        'accuracy': []
        }

    for current_fix_g_option in model_var_config.fix_length_gs:
        for current_fix_f_option in model_var_config.fix_length_fs:            
            for seed1 in all_seeds:
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

                                
                model_preds = []
                all_targets = []

                with torch.no_grad():
                    for current_data in use_dataloader:
                        imgs, current_targets = current_data
                        model_preds.append(model1.predict_targets(imgs, device))
                        all_targets.append(current_targets)
                
                model_preds = torch.concatenate(model_preds, dim = 0)
                all_targets = torch.concatenate(all_targets, dim = 0).detach().cpu()

                if len(all_targets.shape) != 1:
                    all_targets_int = torch.tensor(convert_one_hot_to_ints(all_targets))
                else:
                    all_targets_int = all_targets

                acc = (model_preds == all_targets_int).sum()/all_targets_int.shape[0]
                if (current_fix_g_option == 0) and (current_fix_f_option == 0):
                    print(f'Accuracy layer size: {layer_size}, seed {seed1}: {acc}')
                else:
                    print(f'Accuracy layer size: {layer_size}, fix_g {current_fix_g_option}, fix_f {current_fix_f_option}, seed {seed1}: {acc}')

                accuracy_dict['model_seed'].append(seed1)
                accuracy_dict['fix_g'].append(current_fix_g_option)
                accuracy_dict['fix_f'].append(current_fix_f_option)
                accuracy_dict['num_classes'].append(num_classes)
                accuracy_dict['fd'].append(rep_dim)
                accuracy_dict['accuracy'].append(acc)
            
            
    if model_type == 'SmallMLP':
        file_name = f'{date_str}_model_acc{data_suff}_{model_type}{size_suff}_cls{num_classes}{extra_suff}.json'
    elif model_type in ('SmallCIFAR10', 'MedCIFAR10'):
        file_name = f'{date_str}_model_acc{data_suff}_{model_type}{size_suff}_fd{model_var_config.rep_dim}{step_suff}.json' 
    else:
        raise ValueError(f'Did not recognize model type: {model_type}') 

    result_folder = 'results'
    file_path = os.path.join(result_folder, file_name)

    save_as_json(accuracy_dict, file_path)


def get_and_save_all_synth_accuracies():
    """ Get and save accuracies for all model son synthetic data """
    num_classes = [4, 6, 10, 18]
    date_strings = ['2025-04-22', '2024-12-06', '2024-12-06', '2025-04-22']
    layer_sizes = [16, 32, 64, 128, 256]
    device = 'cpu'

    for i, current_class_num in enumerate(num_classes):
        current_date_str = date_strings[i]
        for current_layer_size in layer_sizes:
            get_and_save_accuracies(
                current_date_str, current_layer_size, current_class_num, 
                '', device, use_train_for_acc=False,
                    model_var_config_path='', dataset_config_path='',
                    train_config_path='' )
            