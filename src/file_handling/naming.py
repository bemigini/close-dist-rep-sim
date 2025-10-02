"""

For making consistent filenames

"""



from datetime import datetime
import os


from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelConfig
from src.config.train_config import TrainConfig


def get_model_variations_config_name():
    """ Get the file name of the model variations config """
    return 'model_variations_config.json'


def get_dataset_config_name(dataset_config: DatasetConfig) -> str:
    """ Get dataset config filename"""
    d_name = dataset_config.dataset_name
    n_points = dataset_config.num_points
    seed = dataset_config.random_seed
    num_cls = dataset_config.num_classes
    if dataset_config.equal_weighting_classes:
        eq_weight = '_eqw'
    else:
        eq_weight = ''
    if dataset_config.min_length > 0:
        min_len = f'_minlen{dataset_config.min_length}'
    else:
        min_len = ''
    if dataset_config.dist_scale > 0:
        d_scale = f'_dscale{dataset_config.dist_scale}'
    else:
        d_scale = ''
    if dataset_config.vert_dist_scale > 0:
        v_scale = f'_vscale{dataset_config.vert_dist_scale}'
    else:
        v_scale = ''

    file = f'{d_name}_{n_points}_{seed}_cls{num_cls}{eq_weight}{min_len}{d_scale}{v_scale}.json'
    return file 


def get_nn_config_suffix(config: ModelConfig) -> str:
    
    suffix = f'{config.model_type}_{config.target_type.name}_{config.nonlinearity}_{config.random_seed}'
    
    return suffix


def get_train_config_suffix(config: TrainConfig) -> str:
    """ Get train config file suffix """     
    lr_str = str(config.learning_rate).replace(".", "_")
    suffix = f'{config.dataset_name}_{config.random_seed}_{config.batch_size}_{config.optimizer_type.name}_{lr_str}'
    
    return suffix


def get_model_name(nn_config: ModelConfig, 
                   train_config: TrainConfig) -> str:
    model_config_suffix = get_nn_config_suffix(nn_config)
    train_config_suffix = get_train_config_suffix(train_config)
    model_name = f'{model_config_suffix}_{train_config_suffix}'
    
    return model_name


def get_file_prefix(
        nn_config: ModelConfig, 
        train_config: TrainConfig,
        date_str: str) -> str:
    model_name = get_model_name(nn_config, train_config)
    prefix = f'{date_str}_{model_name}_'
    
    return prefix


def get_file_prefix_from_name(
        model_name: str,
        date_str: str) -> str:   
    prefix = f'{date_str}_{model_name}_'
    
    return prefix


def get_sample_file_prefix(
        file_prefix: str,
        step: int,
        batch_size: int) -> str:
    prefix = f'{file_prefix}{step}_samples_{batch_size}'
    
    return prefix


def get_metrics_path(
        m_config: ModelConfig, 
        train_config: TrainConfig,
        date: str = '') -> str:
    metrics_prefix = get_model_name(m_config, train_config)
    if date == '':
        date = datetime.today().strftime('%Y-%m-%d')
    metrics_file = f'{date}_{metrics_prefix}_metrics.json'
    metrics_path = os.path.join('metrics', metrics_file)
    
    return metrics_path


def get_model_file_name(
        m_config: ModelConfig, 
        train_config: TrainConfig,
        date: str = '') -> str:
    model_prefix = get_model_name(m_config, train_config)
    if date == '':
        date = datetime.today().strftime('%Y-%m-%d')
    file = f'{date}_{model_prefix}'
    
    return file


def get_trained_model_name(
    dataset_name: str,
    n_classes: int, seed: int, num_features: int,
    f_dim: int, num_data_points:int, lr:float, num_steps: int,
    model_type: str, fix_gs: float, fix_fs: float) -> str:
    """ Get the file name of a trained Roeder model """
    if fix_gs > 0:
        g_suff = f'_fixg{str(fix_gs).replace(".", "_")}'
    else:
        g_suff = ''
    
    if fix_fs > 0:
        f_suff = f'_fixf{str(fix_fs).replace(".", "_")}'
    else:
        f_suff = ''
    
    lr = str(lr).replace('.', '_')
    return f'trained_{dataset_name}_{model_type}{f_suff}{g_suff}_classes{n_classes}_seed{seed}_features{num_features}_fd{f_dim}_data{num_data_points}_lr{lr}_steps{num_steps}'


def get_roeder_model_name(
    dataset_name: str,
    n_classes: int, seed: int, num_features: int,
    f_dim: int, num_data_points:int, lr:float, num_steps: int,
    alt_suff: str, alt_loss_cos: bool, alt_loss_batch_cos: bool, 
    alt_loss_KL: bool, alt_loss_length_reg: bool, alt_loss_arccos: bool,
    alt_loss_class: bool, fix_gs: bool, shuffle_fixed_gs: bool, 
    scale_fixed_gs: float, fix_length_gs: bool) -> str:
    """ Get the file name of a trained Roeder model """
    if alt_loss_arccos:
        alt_loss_suff = '_arccosloss'
    elif alt_loss_cos:
        alt_loss_suff = '_cosloss'
    elif alt_loss_batch_cos:
        alt_loss_suff = '_batchcosloss'
    elif alt_loss_KL:
        alt_loss_suff = '_KLloss'
    elif alt_loss_length_reg:
        alt_loss_suff = '_lengthregloss'
    elif alt_loss_class:
        alt_loss_suff = '_classloss'
    else:
        alt_loss_suff = ''
    
    scale_fixed_gs_str = str(scale_fixed_gs).replace(".", "_")

    if fix_gs:
        if shuffle_fixed_gs:
            g_suff = '_shuff_fixg'
        else:
            g_suff = '_fixg'
        if scale_fixed_gs != 1:
            g_suff = f'_{scale_fixed_gs_str}{g_suff}'
    else:
        if fix_length_gs:
            g_suff = f'g_length_{scale_fixed_gs_str}'
        else:
            g_suff = ''
    lr = str(lr).replace('.', '_')
    return f'roeder_{dataset_name}_{alt_suff}mlp{alt_loss_suff}{g_suff}_classes{n_classes}_seed{seed}_features{num_features}_fd{f_dim}_data{num_data_points}_lr{lr}_steps{num_steps}'


def get_untrained_roeder_model_name(
    dataset_name: str,
    n_classes: int, seed: int, num_features: int, f_dim: int, alt_suff: str, 
    fix_gs: bool) -> str:
    """ Get the file name of an untrained Roeder model """
    if fix_gs:
        g_suff = '_fixg'
    else:
        g_suff = ''
    return f'untrained_roeder_{dataset_name}_{alt_suff}mlp{g_suff}_classes{n_classes}_seed{seed}_features{num_features}_fd{f_dim}'


def get_untrained_model_name(
    dataset_name: str,
    n_classes: int, seed: int, num_features: int, f_dim: int, 
    model_type: str, 
    fix_gs: float, fix_fs: float) -> str:
    """ Get the file name of an untrained model """
    if fix_gs > 0:
        g_suff = f'_fixg{str(fix_gs).replace(".", "_")}'
    else:
        g_suff = ''
    
    if fix_fs > 0:
        f_suff = f'_fixf{str(fix_fs).replace(".", "_")}'
    else:
        f_suff = ''
    
    return f'untrained_{dataset_name}_{model_type}{f_suff}{g_suff}_classes{n_classes}_seed{seed}_features{num_features}_fd{f_dim}'


def get_samples_h5_dataset_name():
    """ get the h5_dataset_name constant """
    return 'samples'


def get_polar_model_name(
    dataset_name: str,
    n_classes: int, seed: int, num_features: int,
    f_dim: int, num_data_points:int, lr:float, num_steps: int,
    weight_decay: float, 
    enforce_g_use_circle: bool, fix_gs: bool) -> str:
    """ Get the file name of a trained Polar classifier model """    
    lr = str(lr).replace('.', '_')
    if weight_decay > 0:
        wd_suff = '_wd'+str(weight_decay).replace('.', '_')
    else:
        wd_suff = ''
    if enforce_g_use_circle:
        g_suff = '_gcirc'
    elif fix_gs:
        g_suff = '_fixg'
    else:
        g_suff = ''
    return f'polar{g_suff}_{dataset_name}_mlp_classes{n_classes}_seed{seed}_features{num_features}_fd{f_dim}_data{num_data_points}_lr{lr}{wd_suff}_steps{num_steps}'


def get_untrained_polar_model_name(
    dataset_name: str,
    n_classes: int, seed: int, num_features: int, f_dim: int,
    enforce_g_use_circle: bool, fix_gs: bool) -> str:
    """ Get the file name of an untrained Roeder model """
    if enforce_g_use_circle:
        g_suff = '_gcirc'
    elif fix_gs:
        g_suff = '_fixg'
    else:
        g_suff = ''
    return f'untrained_polar{g_suff}_{dataset_name}_mlp_classes{n_classes}_seed{seed}_features{num_features}_fd{f_dim}'
