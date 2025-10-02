"""


PLot showing mean d_LLV vs width of the network used in the trained model



"""





import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from tqdm import tqdm 

from plots.util import get_dpi, get_figure_folder

from src.file_handling.save_load_json import load_json


def make_combined_dist_and_rep_vs_width_plot(
        data_df, classes_to_use, acc_dict, 
        seed_vs, fontsize: int, figure_folder: str, 
        comb_file_name:str, dpi:int  
    ) -> None:
    """
    Make a plot showing d_LLV and d_SVD vs width side by side.
    
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6))#, sharey=True
    
    colours_to_use = ['#002347', '#FF5003']
    for j, current_class in enumerate(classes_to_use):
        current_df = data_df[data_df['num_classes'] == current_class]
        
        use_mean_d_probs = []
        use_mean_d_reps = []
        use_std_probs = []
        use_std_reps = []
        use_width_idx = []
        for i, size in enumerate(current_df['width'].values):
            current_accuracies_df = acc_dict[current_class][size]
            current_accuracies_df = current_accuracies_df[(current_accuracies_df['fix_g'] == 0)&(current_accuracies_df['fix_f'] == 0)]
            use_model_seeds = current_accuracies_df['model_seed'][current_accuracies_df['accuracy'] > 0.9]
            if use_model_seeds.shape[0] < 5:
                print(f'Too few high acc model seeds. {current_class}, {size}, num seeds: {use_model_seeds.shape[0]}')
                continue

            print(f'High acc model seeds. {current_class}, {size}, num seeds: {use_model_seeds.shape[0]}')

            use_seed_filter = np.apply_along_axis(np.all, 1, np.isin(seed_vs, use_model_seeds))
            current_d_probs = current_df['all_d_prob'][current_df['width'] == size].values[0]
            use_d_probs = current_d_probs[use_seed_filter]            
            use_mean_d_probs.append(use_d_probs.mean())
            use_std_probs.append(np.std(use_d_probs))
            
            current_d_reps = current_df['all_max_d_rep'][current_df['width'] == size].values[0]
            use_d_reps = current_d_reps[use_seed_filter]            
            use_mean_d_reps.append(use_d_reps.mean())
            use_std_reps.append(np.std(use_d_reps))
            
            use_width_idx.append(i)
        
        use_mean_d_probs = np.array(use_mean_d_probs)
        use_mean_d_reps = np.array(use_mean_d_reps)
        use_std_probs = np.array(use_std_probs)
        use_std_reps = np.array(use_std_reps)
        
        current_width = current_df['width'].values[np.array(use_width_idx)]
        ax1.plot(current_width, use_mean_d_probs, 
                c=colours_to_use[j], linewidth=2.5, label = f'{current_class} classes')
        ax1.fill_between(
            current_width, 
            use_mean_d_probs-use_std_probs, use_mean_d_probs+use_std_probs, 
            alpha = 0.2)
        ax2.plot(current_width, use_mean_d_reps, 
                c=colours_to_use[j], linewidth=2.5, label = f'{current_class} classes')
        ax2.fill_between(
            current_width, 
            use_mean_d_reps-use_std_reps, use_mean_d_reps+use_std_reps, 
            alpha = 0.2)
    
    classes_str = '_'.join([str(c) for c in classes_to_use])    

    ax1.set_xlabel('width', fontsize = fontsize)
    ax2.set_xlabel('width', fontsize = fontsize)
    ax1.set_ylabel(r'mean $d^{\lambda}_{\mathrm{LLV}}$', fontsize = fontsize)
    ax2.set_ylabel(r'mean $d_{\mathbf{f},\mathbf{g}}$', fontsize = fontsize)
    ax2.yaxis.set_label_position("right")
    ax2.legend(fontsize = fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
    fig.tight_layout()
    #fig.show()

    figure_name = comb_file_name.replace('cls.json', f'cls{classes_str}.png')
    figure_path = os.path.join(figure_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()



def plot_mean_d_prob_vs_width_synthetic() -> None:
    """
        consider d_LLV vs network width 
    """
    dpi = get_dpi()
    figure_folder = get_figure_folder()
    if figure_folder not in os.listdir():
        os.mkdir(figure_folder)
    result_folder = 'results'
    model_type = 'SmallMLP'

    fontsize = 18
    #rep_dim = 2
    fix_length_gs = [0, 20]
    fix_length_fs = [0, 20]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    num_classes = [4, 6, 10, 18]
    date_strings = ['2025-04-22', '2024-12-06', '2024-12-06', '2025-04-22']
    layer_sizes = [16, 32, 64, 128, 256]
    
    weight = 0.00001
    weight_suff = f'_w{str(weight).replace(".", "_")}'
    data_suff = '_test' 
    dist_type = 'max'
    sum_or_max = 'max'


    d_prob_class_width_dict = {
        's_vs_s': [],
        'all_d_prob': [],        
        'all_max_d_rep': [],
        'fix_g': [],
        'fix_f': [],
        'num_classes': [],
        'width': []
        }
    acc_df_dict = {}

    
    for i, current_num_classes in tqdm(enumerate(num_classes)):
        date_str = date_strings[i]
        acc_df_dict[current_num_classes] = {}
        for current_layer_size in layer_sizes:
            size_suff = f'_{current_layer_size}'

            file_name = f'{date_str}_model_acc{data_suff}_{model_type}{size_suff}_cls{current_num_classes}.json'
            result_folder = 'results'
            file_path = os.path.join(result_folder, file_name)
            acc_json_dict = load_json(file_path)
            acc_df = pd.DataFrame.from_dict(acc_json_dict)   
            acc_df_dict[current_num_classes][current_layer_size] = acc_df

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
                    max_d_reps = np.maximum(1 - min_m_SVDs_g, 1 - min_m_SVDs_f)                    

                    d_prob_class_width_dict['s_vs_s'].append(distance_df['model_seeds'].values)
                    d_prob_class_width_dict['all_d_prob'].append(distances)
                    d_prob_class_width_dict['all_max_d_rep'].append(max_d_reps)
                    d_prob_class_width_dict['fix_g'].append(current_fix_g_option)
                    d_prob_class_width_dict['fix_f'].append(current_fix_f_option)
                    d_prob_class_width_dict['num_classes'].append(current_num_classes)
                    d_prob_class_width_dict['width'].append(current_layer_size)
                    

    df = pd.DataFrame.from_dict(d_prob_class_width_dict)

    df_fix_f_g_0 = df[(df['fix_g'] == 0)&(df['fix_f'] == 0)]

    # plot including only distances between high accuracy models 
    seed_vs = [[int(t) for t in s.split('vs')] for s in df_fix_f_g_0['s_vs_s'][0]]

    # better than 90%

    # Combined plots distributional and representational similarity side by side. 
    comb_file_name = f'd_prob_d_rep_vs_width{data_suff}_{dist_type}{sm_suff}{weight_suff}_{model_type}_cls.json'
    classes_to_use = [4,6]

    make_combined_dist_and_rep_vs_width_plot(
        data_df=df_fix_f_g_0, classes_to_use=classes_to_use, acc_dict=acc_df_dict,
        seed_vs=seed_vs, fontsize=fontsize, figure_folder=figure_folder,
        comb_file_name=comb_file_name, dpi=dpi
    )

    # For 10 and 18 classes 
    classes_to_use = [10]

    make_combined_dist_and_rep_vs_width_plot(
        data_df=df_fix_f_g_0, classes_to_use=classes_to_use, acc_dict=acc_df_dict,
        seed_vs=seed_vs, fontsize=fontsize, figure_folder=figure_folder,
        comb_file_name=comb_file_name, dpi=dpi
    )

    classes_to_use = [18]

    make_combined_dist_and_rep_vs_width_plot(
        data_df=df_fix_f_g_0, classes_to_use=classes_to_use, acc_dict=acc_df_dict,
        seed_vs=seed_vs, fontsize=fontsize, figure_folder=figure_folder,
        comb_file_name=comb_file_name, dpi=dpi
    )
