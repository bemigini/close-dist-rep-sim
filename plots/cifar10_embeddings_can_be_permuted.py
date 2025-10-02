"""


Make plot which shows that embeddings from model on CIFAR10 can be permuted. 



"""




import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import CCA

import torch
from tqdm import tqdm 

from plots.util import get_dpi, get_figure_folder

from src.config.dataset_config import DatasetConfig
from src.config.model_config import ModelVariationsConfig, ModelConfig
from src.config.train_config import TrainConfig
from src.data.data_init import initialize_dataset

import src.dissimilarity_measures.distribution_distance as dd

from src.file_handling.save_load_json import load_json

from src.file_handling.save_load_model import load_trained_model


from src.data.cifar10_data import load_cifar10



def cifar_embs_can_permute_plots():
    """
    Plots showing the embedding representations of resnet models trained on CIFAR-10

    """
    dpi = get_dpi()
    figure_folder = get_figure_folder()
    if figure_folder not in os.listdir():
        os.mkdir(figure_folder)
    fontsize = 18
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    date_str = '2025-04-22'
    layer_size = 128 
    num_classes = 10
    device = 'cpu'
    final_dim = 2

        
    # pylint: disable=no-member
    # both Paired and tab10 exist as colormaps 
    colors = [plt.cm.Paired(i) for i in range(num_classes)]    
    #colors = [plt.cm.tab10(i) for i in range(num_classes)]

    use_train = False        
    
    model_var_config_path = f'configs/model_variations_config_resnetcifar10_128_fd{final_dim}.json'
    model_json_dict = load_json(model_var_config_path)
    model_var_config = ModelVariationsConfig(**model_json_dict)

    all_seeds = model_var_config.random_seeds

    dataset_config_path = 'configs/cifar10_0_cls10.json'
    dataset_json_dict = load_json(dataset_config_path)
    dataset_config = DatasetConfig(**dataset_json_dict)
    
    batch_size = 16
    train_dataloader, test_dataloader = initialize_dataset(
        dataset_config, batch_size=batch_size, shuffle_train=False)

    if use_train:
        use_dataloader = train_dataloader
    else:
        use_dataloader = test_dataloader

    _, _, class_strings = load_cifar10(batch_size=batch_size, shuffle_train=False)


    rep_dim = model_var_config.rep_dim
    
    train_config_path = 'configs/cifar10_0_32_ADAM_0_0001_20000steps.json'
    train_json_dict = load_json(train_config_path)
    train_config = TrainConfig(**train_json_dict)

    checkpoint_folder = 'checkpoints'    


    current_fix_f_option = model_var_config.fix_length_gs[0]
    current_fix_g_option = model_var_config.fix_length_fs[0]

    all_embedding_reps = []
    all_unembedding_reps = []
    all_predictions = []
    all_log_likelihoods = []


    for current_seed in all_seeds:
        model_config1 = ModelConfig(
            current_seed, 
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

        possible_targets_oh = model1.possible_targets_oh
        with torch.no_grad():
            g1_reps = model1.get_g_reps(possible_targets_oh)
        
        all_unembedding_reps.append(g1_reps)

        f1_rep_list = []

        with torch.no_grad():
            for current_data in tqdm(use_dataloader):
                imgs, _ = current_data                
                f1_rep_list.append(model1.get_f_reps(imgs))

        f1_reps = torch.concatenate(f1_rep_list, dim=0)
                
        all_embedding_reps.append(f1_reps)

        ln_p1_yks_x = dd.log_likelihood_from_reps(f1_reps, g1_reps)
        all_log_likelihoods.append(ln_p1_yks_x)

        p1_probs = torch.exp(ln_p1_yks_x)
        p1_preds = torch.argmax(p1_probs, dim = 0)
        all_predictions.append(p1_preds)

    # Make single plots
    for current_seed in all_seeds:
        file_name = f'resnet_cifar10_seed{current_seed}_reps.json'

        f_reps = all_embedding_reps[current_seed]
        g_reps = all_unembedding_reps[current_seed]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5))

        for i in range(num_classes):
            current_f1_reps = f_reps[all_predictions[current_seed] == i]
            current_g_reps = g_reps[i]
            ax1.scatter(current_f1_reps[:, 0], current_f1_reps[:, 1], s = 5, color = colors[i])
            ax2.scatter(current_g_reps[0], current_g_reps[1], s = 150, color = colors[i], label=class_strings[i])
        
        ax1.set_title(r'\textsc{embeddings} $\mathbf{f}(\cdot)$', fontsize=fontsize+2)
        ax2.set_title(r'\textsc{unembeddings} $\mathbf{g}(\cdot)$', fontsize=fontsize+2)

        legend = ax2.legend(
            bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False,
            fontsize = fontsize)
        for j in range(num_classes):
            # pylint: disable=protected-access
            legend.legend_handles[j]._sizes = [150]
        ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
        fig.tight_layout()
        #fig.show()

        figure_name = file_name.replace('_reps.json', '_embeddings_unembeddings.png')
        figure_path = os.path.join(figure_folder, figure_name)
        plt.savefig(figure_path, dpi = dpi)
        plt.close()

    # Make comparison plot 
    comparison_seeds = [0, 1]

    f1_reps = all_embedding_reps[comparison_seeds[0]]
    g1_reps = all_unembedding_reps[comparison_seeds[0]]
    p1_preds = all_predictions[comparison_seeds[0]]

    f2_reps = all_embedding_reps[comparison_seeds[1]]
    g2_reps = all_unembedding_reps[comparison_seeds[1]]
    p2_preds = all_predictions[comparison_seeds[1]]

    n_components = rep_dim
    cca = CCA(n_components=n_components, max_iter=1000)
    cca.fit(f1_reps, f2_reps)
    X_c, Y_c = cca.transform(f1_reps, f2_reps)
    corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
    mCCA_corr = np.mean(corrs)
    print(f'm_CCA: {mCCA_corr}')
    # 0.551193608450113

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10.4, 4.6))

    for i in range(num_classes):
        current_f1_reps = f1_reps[p1_preds == i]
        current_f2_reps = f2_reps[p2_preds == i]
        ax1.scatter(current_f1_reps[:, 0], current_f1_reps[:, 1], s = 5, color = colors[i])
        ax2.scatter(current_f2_reps[:, 0], current_f2_reps[:, 1], s = 5, color = colors[i], label=class_strings[i])
    
    ax1.set_xlabel(r'\textsc{embeddings} $\mathbf{f}(\cdot)$', fontsize=fontsize+2)
    ax2.set_xlabel(r"\textsc{embeddings} $\mathbf{f}'(\cdot)$", fontsize=fontsize+2)
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])

    legend = ax2.legend(
        bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False,
        fontsize = fontsize)
    for j in range(num_classes):
        # pylint: disable=protected-access
        legend.legend_handles[j]._sizes = [150]
    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
    fig.tight_layout()
    #fig.show() 
    
    figure_name = f'resnet_cifar10_embedding_comparison_{"_".join([str(s) for s in comparison_seeds])}'
    figure_path = os.path.join(figure_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)        
    plt.close()


    # Extra comparison info
    mean_kl_12 = dd.get_mean_KL_divergence(f1_reps, g1_reps, f2_reps, g2_reps, 'cpu')
    print(mean_kl_12)

    mean_kl_21 = dd.get_mean_KL_divergence(f2_reps, g2_reps, f1_reps, g1_reps, 'cpu')
    print(mean_kl_21)
