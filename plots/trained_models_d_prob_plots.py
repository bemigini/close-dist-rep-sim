"""


Making plots d_rep vs d_prob plots for the trained models



"""



import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from tqdm import tqdm 

from plots.util import get_dpi, get_figure_folder

from src.file_handling.save_load_json import load_json


def plot_prob_distances_synthetic() -> None:
    """
        consider the probability distances 
    """
    dpi = get_dpi()
    figure_folder = get_figure_folder()
    fontsize = 18
    result_folder = 'results'
    model_type = 'SmallMLP'
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    rep_dim = 2
    fix_length_gs = [0, 20]
    fix_length_fs = [0, 20]

    num_classes = [4, 6, 10, 18]
    date_strings = ['2025-04-22', '2024-12-06', '2024-12-06', '2025-04-22']
    layer_sizes = [16, 32, 64, 128]
    
    weight = 0.00001
    weight_suff = f'_w{str(weight).replace(".", "_")}'
    data_suff = '_test' 
    dist_type = 'max'
    sum_or_max = 'max'

    all_d_probs = []
    all_max_d_reps = []

    
    for i, current_num_classes in tqdm(enumerate(num_classes)):
        date_str = date_strings[i]
        for current_layer_size in layer_sizes:
            size_suff = f'_{current_layer_size}'
            for current_fix_g_option in fix_length_gs:
                for current_fix_f_option in fix_length_fs:

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


                    file_name = f'{date_str}_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}{fix_g_suff}{fix_f_suff}{size_suff}_cls{current_num_classes}.json'                    
                    file_path = os.path.join(result_folder, file_name)

                    json_dict = load_json(file_path)
                    m_SVDs_f = json_dict.pop('m_SVDs_f', None)
                    m_SVDs_f = np.array(m_SVDs_f)
                    m_SVDs_g = json_dict.pop('m_SVDs_g', None)
                    m_SVDs_g = np.array(m_SVDs_g)
                    min_m_SVDs_f = np.array([np.min(m) for m in m_SVDs_f])
                    min_m_SVDs_g = np.array([np.min(m) for m in m_SVDs_g])
                    distance_df = pd.DataFrame.from_dict(json_dict)   

                    distances = distance_df['distance']
                    t_1s = distance_df['t_1']
                    t_2s = distance_df['t_2']

                    max_d_reps = np.maximum(1 - min_m_SVDs_g, 1 - min_m_SVDs_f)

                    all_d_probs.extend(distances)
                    all_max_d_reps.extend(max_d_reps)
                                    
                    distances_ord_filter = np.argsort(distances)
                    idx_count = np.arange(distances.shape[0])
                    distances_ord = distances[distances_ord_filter]

                    # plot ordered distances
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(idx_count, distances_ord, s=2)
                    figure_name = file_name.replace('.json', '_distances.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    plt.close()

                    # Plot distances vs min_m_SVDs (both f and g) small distances only
                    max_dist = 1/(2*rep_dim)
                    small_dist_filt = distances < max_dist

                    possible_distances = np.arange(0.0005, max_dist+0.01, step=0.02)
                    bound = 4*possible_distances

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(distances[small_dist_filt], max_d_reps[small_dist_filt])
                    ax.plot(possible_distances, bound, label = 'bound')
                    ax.set_xlabel(r'$d^{\lambda}_{\mathrm{LLV}}$', fontsize = fontsize)
                    ax.set_ylabel(r'$\max d_{\mathrm{SVD}}$', fontsize = fontsize)
                    ax.legend(fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_d_prob_vs_d_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    #fig.show()
                    plt.close()


                    # Plot t_2 vs min_m_SVDs 
                    small_t_2_filt =  t_2s < 0.25

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(t_2s[small_t_2_filt], (1 - min_m_SVDs_f)[small_t_2_filt])
                    ax.plot(possible_distances, bound, label = 'bound')
                    ax.set_xlabel('t_2', fontsize = fontsize)
                    ax.set_ylabel(r'$\max d_{\mathrm{SVD}} \mathbf{f}$', fontsize = fontsize)
                    ax.legend(fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_t_2_vs_f_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    #fig.show()
                    plt.close()


                    # Plot t_1 vs min_m_SVDs 
                    small_t_1_filt =  t_1s < 0.25

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(t_1s[small_t_1_filt], (1 - min_m_SVDs_g)[small_t_1_filt])
                    ax.plot(possible_distances, bound, label = 'bound')
                    ax.set_xlabel('t_1', fontsize = fontsize)
                    ax.set_ylabel(r'$\max d_{\mathrm{SVD}} \mathbf{g}$', fontsize = fontsize)
                    ax.legend(fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_t_1_vs_g_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    plt.close()

    all_d_probs = np.array(all_d_probs)
    all_max_d_reps = np.array(all_max_d_reps) 

    classes_str = '_'.join([str(c) for c in num_classes])

    file_name = f'all_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}_all_cls{classes_str}.json'

    max_dist = 1/(2*rep_dim)
    small_dist_filt = all_d_probs < max_dist

    possible_distances = np.arange(0.0005, max_dist+0.01, step=0.02)
    bound = 4*possible_distances

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_d_probs[small_dist_filt], all_max_d_reps[small_dist_filt], s=5)
    ax.plot(possible_distances, bound, label = 'bound')
    ax.set_xlabel(r'$d^{\lambda}_{\mathrm{LLV}}$', fontsize = fontsize)
    ax.set_ylabel(r'$\max d_{\mathrm{SVD}}$', fontsize = fontsize)
    ax.legend(fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    fig.tight_layout()
    figure_name = file_name.replace('.json', '_d_prob_vs_d_rep.png')
    figure_path = os.path.join(figure_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    #fig.show()
    plt.close()



def plot_prob_distances_cifar() -> None:
    """
        consider the probability distances 
    """
    dpi = get_dpi()
    figure_folder = get_figure_folder()
    result_folder = 'results'
    model_type = 'SmallCIFAR10'

    fontsize = 18
    rep_dims = [2, 3]
    fix_length_gs = [0, 20]
    fix_length_fs = [0, 20]

    date_str = '2025-04-22'
    layer_sizes = [128]
    
    weight = 0.00001
    weight_suff = f'_w{str(weight).replace(".", "_")}'    
    dist_type = 'max'
    sum_or_max = 'max'
    data_suff = '_test' 
    
    for current_rep_dim in rep_dims:
        all_d_probs = []
        all_max_d_reps = []
        for current_layer_size in layer_sizes:
            size_suff = f'_{current_layer_size}'
            for current_fix_g_option in fix_length_gs:
                for current_fix_f_option in fix_length_fs:

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


                    file_name = f'{date_str}_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}{fix_g_suff}{fix_f_suff}{size_suff}_fd{current_rep_dim}.json'                    
                    file_path = os.path.join(result_folder, file_name)

                    json_dict = load_json(file_path)
                    m_SVDs_f = json_dict.pop('m_SVDs_f', None)
                    m_SVDs_f = np.array(m_SVDs_f)
                    m_SVDs_g = json_dict.pop('m_SVDs_g', None)
                    m_SVDs_g = np.array(m_SVDs_g)
                    min_m_SVDs_f = np.array([np.min(m) for m in m_SVDs_f])
                    min_m_SVDs_g = np.array([np.min(m) for m in m_SVDs_g])
                    distance_df = pd.DataFrame.from_dict(json_dict)   

                    distances = distance_df['distance']
                    t_1s = distance_df['t_1']
                    t_2s = distance_df['t_2']

                    max_d_reps = np.maximum(1 - min_m_SVDs_g, 1 - min_m_SVDs_f)

                    all_d_probs.extend(distances)
                    all_max_d_reps.extend(max_d_reps)
                                    
                    distances_ord_filter = np.argsort(distances)
                    idx_count = np.arange(distances.shape[0])
                    distances_ord = distances[distances_ord_filter]

                    # plot ordered distances
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(idx_count, distances_ord, s=2)
                    figure_name = file_name.replace('.json', '_distances.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    plt.close()

                    # Plot distances vs min_m_SVDs (both f and g) small distances only
                    max_dist = 1/(2*current_rep_dim)
                    small_dist_filt = distances < max_dist

                    possible_distances = np.arange(0.0005, max_dist+0.01, step=0.02)
                    bound = 4*possible_distances

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(distances[small_dist_filt], max_d_reps[small_dist_filt])
                    ax.plot(possible_distances, bound, label = 'bound')
                    ax.set_xlabel(r'$d^{\lambda}_{\mathrm{LLV}}$', fontsize = fontsize)
                    ax.set_ylabel(r'$\max d_{\mathrm{SVD}}$', fontsize = fontsize)
                    ax.legend(fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_d_prob_vs_d_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    #fig.show()
                    plt.close()


                    # Plot t_2 vs min_m_SVDs 
                    small_t_2_filt =  t_2s < 0.25

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(t_2s[small_t_2_filt], (1 - min_m_SVDs_f)[small_t_2_filt])
                    ax.plot(possible_distances, bound, label = 'bound')
                    ax.set_xlabel('t_2', fontsize = fontsize)
                    ax.set_ylabel('max d_rep f', fontsize = fontsize)
                    ax.legend(fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_t_2_vs_f_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    #fig.show()
                    plt.close()


                    # Plot t_1 vs min_m_SVDs 
                    small_t_1_filt =  t_1s < 0.25

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(t_1s[small_t_1_filt], (1 - min_m_SVDs_g)[small_t_1_filt])
                    ax.plot(possible_distances, bound, label = 'bound')
                    ax.set_xlabel('t_1', fontsize = fontsize)
                    ax.set_ylabel('max d_rep g', fontsize = fontsize)
                    ax.legend(fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_t_1_vs_g_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    plt.savefig(figure_path, dpi = dpi)
                    plt.close()

        all_d_probs = np.array(all_d_probs)
        all_max_d_reps = np.array(all_max_d_reps) 

        file_name = f'all_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}_fd{current_rep_dim}_all.json'

        max_dist = 1/(2*current_rep_dim)
        small_dist_filt = all_d_probs < max_dist

        possible_distances = np.arange(0.0005, max_dist+0.01, step=0.02)
        bound = 4*possible_distances

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_d_probs[small_dist_filt], all_max_d_reps[small_dist_filt], s=12)
        ax.plot(possible_distances, bound, label = 'bound')
        ax.set_xlabel(r'$d^{\lambda}_{\mathrm{LLV}}$', fontsize = fontsize)
        ax.set_ylabel(r'$\max d_{\mathrm{SVD}}$', fontsize = fontsize)
        ax.legend(fontsize = fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        fig.tight_layout()
        figure_name = file_name.replace('.json', '_d_prob_vs_d_rep.png')
        figure_path = os.path.join(figure_folder, figure_name)
        plt.savefig(figure_path, dpi = dpi)
        #fig.show()
        plt.close()
