"""

For training models

"""

import logging 
import os
from typing import List

import numpy as np

import torch

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelConfig
from src.config.train_config import TrainConfig

from src.data.data_init import initialize_dataset

import src.training.train as tr

from src.file_handling.save_load_json import save_as_json
import src.file_handling.naming as name 
from src.file_handling import save_load_model

from src.util import convert_one_hot_to_ints, convert_int_targets_to_one_hot

def get_checkpoint_folder():
    """ Get the name of the folder used for checkpoints """
    return 'checkpoints'
    

def get_metrics_folder():
    """ Get the name of the folder used for metrics """
    return 'metrics'


def train_models(
    model_seeds: List[int],
    model_config: ModelConfig,
    train_config: TrainConfig,
    dataset_config: DatasetConfig,
    date_str_use: str, 
    continue_from_step: int, 
    checkpoint_folder: str, metrics_folder: str,
    device: str):
    """ Train models for the given seeds"""
    if checkpoint_folder == '':
        checkpoint_folder = get_checkpoint_folder()
    if metrics_folder == '':
        metrics_folder = get_metrics_folder()

    train_loader, test_loader = initialize_dataset(dataset_config, train_config.batch_size)

    for current_seed in model_seeds:

        torch.manual_seed(current_seed)

        trained_model_file_name = name.get_trained_model_name(
            train_config.dataset_name,
            model_config.num_classes, current_seed, 
            model_config.num_features, 
            model_config.rep_dim, dataset_config.num_points, 
            train_config.learning_rate, train_config.train_steps, 
            model_config.model_type,
            fix_fs=model_config.fix_length_fs, fix_gs=model_config.fix_length_gs)
        trained_model_file_name = date_str_use + trained_model_file_name
        trained_file_path = os.path.join(checkpoint_folder, trained_model_file_name)

        if os.path.isfile(trained_file_path):
            logging.info('Model already exists, seed: %s', trained_file_path)
            continue

        if continue_from_step > 0:
            load_model_file_name = name.get_trained_model_name(
                train_config.dataset_name,
                model_config.num_classes, current_seed, 
                model_config.num_features, 
                model_config.rep_dim, dataset_config.num_points, 
                train_config.learning_rate, continue_from_step, 
                model_config.model_type,
                fix_fs=model_config.fix_length_fs, fix_gs=model_config.fix_length_gs)
            load_model_file_name = date_str_use + load_model_file_name
            checkpoint_file_path = os.path.join(checkpoint_folder, load_model_file_name)

            if not os.path.isfile(checkpoint_file_path):
                raise ValueError(f'Model not found: {checkpoint_file_path}')
            
            logging.info('Loading model %s', checkpoint_file_path)
            current_model = save_load_model.load_trained_model(
                model_config, train_config, dataset_config, 
                date_str_use, current_seed, checkpoint_folder,
                device=device, overwrite_train_steps=continue_from_step)
            train_steps = train_config.train_steps - continue_from_step

        else:
            current_model = save_load_model.get_untrained_model(
                model_config, train_config, date_str_use, current_seed, checkpoint_folder)

            train_steps = train_config.train_steps
        

        train_losses, test_losses = tr.train(
                    train_loader, test_loader,
                    model_config.target_type,
                    current_model, train_config, device,
                    train_steps)
        print(f'Final train loss: {train_losses[-1]}')
        print(f'Final test loss: {test_losses[-1]}')

        metric_dict = {
            'train_loss': train_losses,
            'test_loss': test_losses}
        metrics_path = os.path.join(metrics_folder, trained_model_file_name + '_metrics.json')

        save_as_json(metric_dict, metrics_path)

        torch.save(current_model.state_dict(), trained_file_path)

        all_test_targets = []
        all_final_preds = [] 
        for current_test_data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = current_test_data
            if len(labels.shape) == 1:
                labels = convert_int_targets_to_one_hot(labels, current_model.num_targets)

            current_inputs = inputs.to(device)
            with torch.no_grad():
                current_preds = current_model.predict_targets(current_inputs, device)

            current_labels = labels.detach().cpu()
            if len(current_labels.shape) != 1:
                current_int_labels = convert_one_hot_to_ints(current_labels)
            else:
                current_int_labels = current_labels
            all_test_targets.extend(current_int_labels)
            all_final_preds.extend(current_preds.detach().cpu().numpy())
                
        all_test_targets = np.array(all_test_targets)
        all_final_preds = np.array(all_final_preds)
        
        correct_final_preds = np.sum(all_test_targets == all_final_preds)
        final_accuracy = correct_final_preds/len(all_test_targets) 
        print(f'Final test accuracy: {final_accuracy}')
