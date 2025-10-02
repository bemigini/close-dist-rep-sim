"""
Run Script for training classification models and getting distances for identifiability experiments


Usage:
    run.py train-variations --output-folder=<file> --dataset-config=<file> --model-var-config=<file> --train-config=<file> [options]
    run.py train-variations --output-folder=<file> --dataset-config=<file> --model-var-config=<file> --train-config=<file> --date-str=<string> [options]
    run.py get-distances --date-str=<string> --layer-size=<List> --num-classes=<int> --dist-type=<str> [options]
    run.py get-distances --date-str=<string> --dataset-config=<file> --model-var-config=<file> --train-config=<file> --layer-size=<List> --num-classes=<int> --dist-type=<str> [options]
    run.py get-accuracies --date-str=<string> --layer-size=<List> --num-classes=<int> [options]
    run.py make-plot --plot-type=<string> [options]
    
Options:
    -h --help                               show this screen.
    --output-folder=<file>                  folder to save trained model(s) in
    --data-folder=<file>                    folder to load data from  
    --dataset-config=<file>                 config to use for dataset 
    --log=<string>                          log level to use [default: info]  
    --model-config=<file>                   config to use for model initialization
    --model-var-config=<file>               config to use for training model variations
    --train-config=<file>                   config to use for training
    --train-seed=<int>                      if set, will overwrite the seed in the training config.[default: -1]    
    --train-epochs=<int>                    If set, will overwrite the epochs in the training config.[default: 0]    
    --lr=<float>                            If set, will overwrite learning rate in training config [default: -1]
    --date-str=<string>                     Add date string to use in model name. If empty will use today's date.
    --weight-decay=<float>                  Set weight decay [default: 0]
    --layer-size=<List>                     number of neurons in the model layers. Separate with commas.
    --num-classes=<int>                     number of classes in the classification problem 
    --num-samples=<int>                     number of samples to use for estimation [default: 10]
    --weight=<float>                        how to weight the last two terms of the distribution distance [default: 1]
    --extra-suff=<str>                      extra suffix to add the the variations config file 
    --dist-type=<str>                       the distance type to use, max or mean
    --sum-or-max=<str>                      whether to use sum or max between term 1 and 2 in the distance [default: sum]
    --continue-from=<int>                   continue from checkpoint with number of steps [default: 0]
    --plot-type=<string>                    the type of the plot to make. Can be: 'cifar10_reps', 'd_LLV_constructed', 'd_LLV_train_synthetic', 'd_LLV_vs_width', 'synthetic_data' or 'KL_table'   
    --use-train                             use training set to calculate distances. If not set will use test set.
    --cuda                                  use GPU 
    
"""



from datetime import datetime 

import logging

from typing import Dict

from docopt import docopt

import plots

import plots.cifar10_embeddings_can_be_permuted
import plots.constructed_models_d_prob_plots
import plots.loss_diff_vs_mcca
import plots.mean_d_prob_vs_width
import plots.trained_models_d_prob_plots
import plots.training_data_illustration

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelConfig, ModelVariationsConfig
from src.config.train_config import TrainConfig

from src.file_handling.save_load_json import load_json

from src.training.training_models import train_models
from src.util import TargetType

from experiments import prob_distances
from experiments import get_and_save_model_accuracies
from experiments import kl_to_zero_dissimilar_reps



def train_variations(args:Dict) -> None:
    """
    Train model variations 
    """
    # Load in options    
    model_var_config_path = args['--model-var-config'] if args['--model-var-config'] else ''
    train_config_path = args['--train-config'] if args['--train-config'] else ''
    dataset_config_path = args['--dataset-config'] if args['--dataset-config'] else '.'
    
    if train_config_path == '' or model_var_config_path == '' or dataset_config_path == '':
        raise ValueError('Train, model and dataset config paths must be given')
    
    continue_from_step = int(args['--continue-from']) if args['--continue-from'] else 0

    output_folder = args['--output-folder'] if args['--output-folder'] else '.'

    date_str = args['--date-str'] if args['--date-str'] else ''  
    device = 'cuda' if args['--cuda'] else 'cpu'
    logging.info('The available device is %s', device)
    logging.info('train_models')

    overwrites = {}
    train_seed = int(args['--train-seed']) if args['--train-seed'] else -1    
    epochs = int(args['--train-epochs']) if args['--train-epochs'] else 0
    lr = float(args['--lr']) if args['--lr'] else -1
    weight_decay = float(args['--weight-decay']) if args['--weight-decay'] else 0.

    if train_seed > -1:
        overwrites['train_seed'] = train_seed    
    if epochs > 0:
        overwrites['num_epochs'] = epochs
    if lr > 0:
        overwrites['lr'] = lr
    if weight_decay > 0:
        overwrites['weight_decay'] = weight_decay        

    # Load train config from json
    train_json_dict = load_json(train_config_path)
    if 'train_seed' in overwrites:
        train_json_dict['train_seed'] = overwrites['train_seed']
    if 'num_epochs' in overwrites:
        train_json_dict['num_epochs'] = overwrites['num_epochs']
    if 'lr' in overwrites:
        train_json_dict['lr'] = overwrites['lr']
    if 'weight_decay' in overwrites:
        train_json_dict['weight_decay'] = overwrites['weight_decay']

    train_config = TrainConfig(**train_json_dict)

    dataset_json_dict = load_json(dataset_config_path)
    dataset_config = DatasetConfig(**dataset_json_dict)

    # Load model variation config
    model_json_dict = load_json(model_var_config_path)
    model_var_config = ModelVariationsConfig(**model_json_dict)

    for current_num_features in model_var_config.num_features:
        for current_length_gs in model_var_config.fix_length_gs:
            for current_length_fs in model_var_config.fix_length_fs:
                model_config = ModelConfig(
                    model_var_config.random_seeds[0], 
                    model_var_config.model_type, 
                    TargetType[model_var_config.target_type], 
                    model_var_config.nonlinearity,
                    model_var_config.num_classes,
                    current_num_features,
                    model_var_config.rep_dim,
                    current_length_gs,
                    current_length_fs
                )
    
                if date_str != '':
                    use_date = date_str
                else:
                    use_date = datetime.today().strftime('%Y-%m-%d')
                
                train_models(
                    model_var_config.random_seeds,
                    model_config, train_config, dataset_config,
                    use_date, continue_from_step=continue_from_step, 
                    checkpoint_folder=output_folder, 
                    metrics_folder='', device=device)
    
    logging.info('train_models end')


def get_distances(args:Dict) -> None:
    """ Get and save distances between model distributions """

    date_str = args['--date-str'] if args['--date-str'] else ''
    layer_sizes = args['--layer-size'] if args['--layer-size'] else '32'
    layer_sizes = [int(s) for s in layer_sizes.split(',')]

    num_classes = int(args['--num-classes']) if args['--num-classes'] else 6
    num_samples = int(args['--num-samples']) if args['--num-samples'] else 20
    weight = float(args['--weight']) if args['--weight'] else 1
    use_train = True if args['--use-train'] else False

    extra_suff = args['--extra-suff'] if args['--extra-suff'] else ''
    dist_type = args['--dist-type'] if args['--dist-type'] else ''
    sum_or_max = args['--sum-or-max'] if args['--sum-or-max'] else 'sum'

    if dist_type not in ['max', 'mean']:
        raise ValueError(f'Unknown distance type: {dist_type}')
    
    model_config_path = args['--model-var-config'] if args['--model-var-config'] else ''
    train_config_path = args['--train-config'] if args['--train-config'] else ''
    dataset_config_path = args['--dataset-config'] if args['--dataset-config'] else ''

    device = 'cuda' if args['--cuda'] else 'cpu'

    logging.info('Device: %s', device)
    logging.info('get_d_prob_between_models')

    for layer_size in layer_sizes:
        prob_distances.get_d_prob_between_models(
            date_str, layer_size, num_classes, extra_suff, 
            weight, device, dist_type, sum_or_max, 
            num_samples=num_samples, use_train_for_distances=use_train,
            model_var_config_path=model_config_path, 
            dataset_config_path = dataset_config_path,
            train_config_path=train_config_path)
    
    logging.info('get_d_prob_between_models end')


def get_accuracies(args:Dict) -> None:
    """ Get and save accuracies of models """

    date_str = args['--date-str'] if args['--date-str'] else ''
    layer_sizes = args['--layer-size'] if args['--layer-size'] else '32'
    layer_sizes = [int(s) for s in layer_sizes.split(',')]

    num_classes = int(args['--num-classes']) if args['--num-classes'] else 6
    use_train = True if args['--use-train'] else False

    extra_suff = args['--extra-suff'] if args['--extra-suff'] else ''
    
    model_config_path = args['--model-var-config'] if args['--model-var-config'] else ''
    train_config_path = args['--train-config'] if args['--train-config'] else ''
    dataset_config_path = args['--dataset-config'] if args['--dataset-config'] else ''

    device = 'cuda' if args['--cuda'] else 'cpu'

    logging.info('Device: %s', device)
    logging.info('get_and_save_accuracies')

    for current_layer_size in layer_sizes:
        get_and_save_model_accuracies.get_and_save_accuracies(
            date_str, current_layer_size, num_classes, extra_suff, device, 
            use_train_for_acc=use_train, model_var_config_path=model_config_path,
            dataset_config_path=dataset_config_path, train_config_path=train_config_path)
    
    logging.info('get_and_save_accuracies end')


def make_plot(args:Dict) -> None:
    """ Make and save plots of the indicated type """
    plot_type = args['--plot-type'] if args['--plot-type'] else ''

    if plot_type == '':
        raise ValueError('Plot type must be given')
    
    logging.info('making plot %s', plot_type)
    
    match plot_type:
        case 'cifar10_reps':
            plots.cifar10_embeddings_can_be_permuted.cifar_embs_can_permute_plots()
        case 'd_LLV_constructed':
            plots.constructed_models_d_prob_plots.plot_constructed_model_examples()
        case 'd_LLV_train_synthetic':
            plots.trained_models_d_prob_plots.plot_prob_distances_synthetic()
        case 'd_LLV_vs_width':
            plots.mean_d_prob_vs_width.plot_mean_d_prob_vs_width_synthetic()
        case 'loss_diff_vs_mcca':
            plots.loss_diff_vs_mcca.plot_all_cifar_loss_diff_vs_mcca()
        case 'synthetic_data':
            num_classes = int(args['--num-classes']) if args['--num-classes'] else 6
            plots.training_data_illustration.get_illustration_synthetic_train_data(
                num_classes=num_classes)
        case 'KL_table':
            device = 'cuda' if args['--cuda'] else 'cpu'
            table_dict = kl_to_zero_dissimilar_reps.kl_to_zero_dissimilar_reps(device=device)
            latex = kl_to_zero_dissimilar_reps.make_latex_table_from_dict(table_dict=table_dict)
            print()
            print(latex)
            print()
        case _:
            raise ValueError(f'Plot type not recognized: {plot_type}')
    
    logging.info('making plot %s end', plot_type)
        

def main():     
    """ Set logging and call relevant function """
    args = docopt(__doc__)
    
    log_level = args['--log'] if args['--log'] else ''
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=numeric_level)   
    
    
    if args['train-variations']:
        train_variations(args)
    elif args['get-distances']:
        get_distances(args)
    elif args['get-accuracies']:
        get_accuracies(args)
    elif args['make-plot']:
        make_plot(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
    