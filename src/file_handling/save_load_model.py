"""

Saving and loading a model


"""


import os

import torch

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelConfig
from src.config.train_config import TrainConfig
from src.file_handling import naming

from src.models.model_class import ModelClass
from src.models.med_cnn import MedCNN
from src.models.small_cnn import SmallCNN
from src.models.small_mlp import SmallMLP
from src.models.resnet import ResNet18




def initialize_model(model_config: ModelConfig) -> torch.nn.Module:
    """ Initialize a model from config """
    match model_config.model_type:
        case 'SmallMLP':
            f_net = SmallMLP(
                input_dim = 2, final_dim = model_config.rep_dim,
                num_features = model_config.num_features, 
                nonlinearity = model_config.nonlinearity)
            g_net = SmallMLP(
                input_dim = model_config.num_classes - 1, final_dim = model_config.rep_dim,
                num_features = model_config.num_features, 
                nonlinearity = model_config.nonlinearity)

            model = ModelClass(
                num_targets = model_config.num_classes,
                embedding_function = f_net,
                unembedding_function = g_net,
                fix_length_gs = model_config.fix_length_gs,
                fix_length_fs = model_config.fix_length_fs)
        case 'SmallCIFAR10':
            f_net = SmallCNN(
                input_channels = 3, final_dim = model_config.rep_dim,
                nonlinearity = model_config.nonlinearity)
            g_net = SmallMLP(
                input_dim = model_config.num_classes - 1, final_dim = model_config.rep_dim,
                num_features = model_config.num_features, 
                nonlinearity = model_config.nonlinearity)

            model = ModelClass(
                num_targets = model_config.num_classes,
                embedding_function = f_net,
                unembedding_function = g_net,
                fix_length_gs = model_config.fix_length_gs,
                fix_length_fs = model_config.fix_length_fs)
        case 'MedCIFAR10':
            f_net = MedCNN(
                input_channels = 3, final_dim = model_config.rep_dim,
                num_nodes = model_config.num_features,
                nonlinearity = model_config.nonlinearity)
            g_net = SmallMLP(
                input_dim = model_config.num_classes - 1, final_dim = model_config.rep_dim,
                num_features = model_config.num_features, 
                nonlinearity = model_config.nonlinearity)

            model = ModelClass(
                num_targets = model_config.num_classes,
                embedding_function = f_net,
                unembedding_function = g_net,
                fix_length_gs = model_config.fix_length_gs,
                fix_length_fs = model_config.fix_length_fs)
        case 'ResNetCIFAR10':
            f_net = ResNet18(final_dim = model_config.rep_dim)
            g_net = SmallMLP(
                input_dim = model_config.num_classes - 1, final_dim = model_config.rep_dim,
                num_features = model_config.num_features, 
                nonlinearity = model_config.nonlinearity)

            model = ModelClass(
                num_targets = model_config.num_classes,
                embedding_function = f_net,
                unembedding_function = g_net,
                fix_length_gs = model_config.fix_length_gs,
                fix_length_fs = model_config.fix_length_fs)
        case _:
            raise ValueError(f'Model type not recognized: {model_config.model_type}')
    
    return model 


def get_untrained_model(
    model_config: ModelConfig, train_config: TrainConfig, 
    date_str_use: str, seed: int, checkpoint_folder: str):
    """
        Get an untrained model
    """
    untrained_model_file_name = naming.get_untrained_model_name(
            train_config.dataset_name,
            model_config.num_classes, seed, 
            model_config.num_features, 
            model_config.rep_dim, model_config.model_type, 
            model_config.fix_length_gs, model_config.fix_length_fs)
    untrained_model_file_name = date_str_use + untrained_model_file_name
    untrained_model_path = os.path.join(checkpoint_folder, untrained_model_file_name)
    
    current_model = initialize_model(model_config)

    if os.path.isfile(untrained_model_path):
        print(f'Untrained model already exists, seed: {seed}')            
        current_model.load_state_dict(torch.load(untrained_model_path, weights_only=True))
    else:            
        torch.save(current_model.state_dict(), untrained_model_path)
        print(f'Saved untrained model seed: {seed}')
    
    return current_model


def load_trained_model(
    model_config: ModelConfig, train_config: TrainConfig,
    dataset_config: DatasetConfig, 
    date_str_use: str, seed: int, checkpoint_folder: str, device: str,
    overwrite_train_steps: int = 0):
    """ 
        Load a trained model 
    """
    train_steps = overwrite_train_steps if overwrite_train_steps > 0 else train_config.train_steps

    model_file_name = naming.get_trained_model_name(
            train_config.dataset_name,
            model_config.num_classes, seed, 
            model_config.num_features, 
            model_config.rep_dim, dataset_config.num_points, 
            train_config.learning_rate, train_steps, 
            model_config.model_type,
            fix_fs=model_config.fix_length_fs, fix_gs=model_config.fix_length_gs)
    model_file_name = date_str_use + model_file_name
    checkpoint_file_path = os.path.join(checkpoint_folder, model_file_name)

    model = initialize_model(model_config)
    model.load_state_dict(
        torch.load(checkpoint_file_path, weights_only=True, map_location=torch.device(device)))

    return model
