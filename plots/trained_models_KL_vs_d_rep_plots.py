"""


Making plots d_rep vs KL plots for the trained models



"""



import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from tqdm import tqdm 

from plots.util import get_dpi, get_figure_folder

from src.file_handling.save_load_json import load_json


def plot_KL_vs_d_rep_synthetic() -> None:
    """
        consider the probability distances 
    """
    dpi = get_dpi()
    figure_folder = get_figure_folder()
    result_folder = 'results'
    model_type = 'SmallMLP'

    fontsize = 18
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

    all_KL = []
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
                    
                    KL_div = distance_df['KL_div']
                    max_d_reps = np.maximum(1 - min_m_SVDs_g, 1 - min_m_SVDs_f)

                    all_KL.extend(KL_div)
                    all_max_d_reps.extend(max_d_reps)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(KL_div, max_d_reps)
                    ax.set_xlabel('KL_div', fontsize = fontsize)
                    ax.set_ylabel('max d_rep', fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_KL_vs_d_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    #fig.show()
                    plt.savefig(figure_path, dpi = dpi)
                    plt.close()


    all_KL = np.array(all_KL)
    all_max_d_reps = np.array(all_max_d_reps) 

    classes_str = '_'.join([str(c) for c in num_classes])

    file_name = f'all_model_distances{data_suff}_KL_{model_type}_all_cls{classes_str}.json'

    max_KL = 0.5
    small_KL_filt = all_KL < max_KL

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_KL, all_max_d_reps, s=5)
    ax.set_xlabel('KL', fontsize = fontsize)
    ax.set_ylabel('max d_rep', fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    fig.tight_layout()
    figure_name = file_name.replace('.json', '_KL_vs_d_rep.png')
    figure_path = os.path.join(figure_folder, figure_name)
    #fig.show()
    plt.savefig(figure_path, dpi = dpi)    
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_KL[small_KL_filt], all_max_d_reps[small_KL_filt], s=5)
    ax.set_xlabel('KL', fontsize = fontsize)
    ax.set_ylabel('max d_rep', fontsize = fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    fig.tight_layout()
    figure_name = file_name.replace('.json', '_small_KL_vs_d_rep.png')
    figure_path = os.path.join(figure_folder, figure_name)
    #fig.show()
    plt.savefig(figure_path, dpi = dpi)    
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
        all_KL = []
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

                    kl_div = distance_df['KL_div']
                    max_d_reps = np.maximum(1 - min_m_SVDs_g, 1 - min_m_SVDs_f)

                    all_KL.extend(kl_div)
                    all_max_d_reps.extend(max_d_reps)

                    # Plot distances vs min_m_SVDs (both f and g) small distances only
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(kl_div, max_d_reps)
                    ax.set_xlabel('KL', fontsize = fontsize)
                    ax.set_ylabel('max d_rep', fontsize = fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                    fig.tight_layout()
                    figure_name = file_name.replace('.json', '_KL_vs_d_rep.png')
                    figure_path = os.path.join(figure_folder, figure_name)
                    #fig.show()
                    plt.savefig(figure_path, dpi = dpi)                    
                    plt.close()


        all_KL = np.array(all_KL)
        all_max_d_reps = np.array(all_max_d_reps) 

        file_name = f'all_model_distances{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}_fd{current_rep_dim}_all.json'

        max_KL = 0.5
        small_kl_filt = all_KL < max_KL

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_KL, all_max_d_reps, s=12)
        ax.set_xlabel('KL', fontsize = fontsize)
        ax.set_ylabel('max d_rep', fontsize = fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        fig.tight_layout()
        figure_name = file_name.replace('.json', '_KL_vs_d_rep.png')
        figure_path = os.path.join(figure_folder, figure_name)
        plt.savefig(figure_path, dpi = dpi)
        #fig.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_KL[small_kl_filt], all_max_d_reps[small_kl_filt], s=12)
        ax.set_xlabel('KL', fontsize = fontsize)
        ax.set_ylabel('max d_rep', fontsize = fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        fig.tight_layout()
        figure_name = file_name.replace('.json', '_small_KL_vs_d_rep.png')
        figure_path = os.path.join(figure_folder, figure_name)
        plt.savefig(figure_path, dpi = dpi)
        #fig.show()
        plt.close()
