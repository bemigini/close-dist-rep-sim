"""


Making plots for the constructed models



"""



import os 

import matplotlib.pyplot as plt
import numpy as np 


from experiments.making_constructed_models import make_even_spaced_classifier_2d, get_distances_to_perturbed_models

from plots.util import get_dpi, get_figure_folder



def plot_constructed_model_examples():
    """ Make constructed models and plot examples of distances """
    fontsize = 18
    figure_folder = get_figure_folder()
    if figure_folder not in os.listdir():
        os.mkdir(figure_folder)
    dpi = get_dpi()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    num_classes = [4, 5, 6, 10, 18]
    noise_levels = [0.005, 0.004, 0.004, 0.0025, 0.002]

    num_inputs_per_class = 500

    for i, current_num_classes in enumerate(num_classes):
        model_name = f'constructed_even_{current_num_classes}_classes_noisy'

        current_noise_level = noise_levels[i]
        m1_all_fx, m1_gy, g_m1_angles, x_angles_m1, fx_lengths = make_even_spaced_classifier_2d(
            num_classes=current_num_classes, num_inputs_per_class=num_inputs_per_class)
        
        dist_bound_dict = get_distances_to_perturbed_models(
            m1_all_fx, m1_gy, g_m1_angles, x_angles_m1, fx_lengths,
            num_classes=current_num_classes, noise_level=current_noise_level
        )

        max_d_rep_f = np.array([1 - np.min(m) for m in dist_bound_dict['msvds_Lfx']])
        max_d_rep_g = np.array([1 - np.min(m) for m in dist_bound_dict['msvds_Ngy']])
        d_prob_array = np.array(dist_bound_dict['d_prob'])

        t_1s = np.array(dist_bound_dict['t_1'])
        t_2s = np.array(dist_bound_dict['t_2'])

        bound_valid_filt = d_prob_array < 0.25
        bound_valid_distances = d_prob_array[bound_valid_filt]
                
        possible_distances = np.arange(0.0005, np.max(bound_valid_distances)+0.01, step=0.02)
        bound = 4*possible_distances

        # bound vs max of embeddings and unembeddings 
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(possible_distances, bound, label = 'bound')
        ax.scatter(bound_valid_distances, np.maximum(max_d_rep_f, max_d_rep_g)[bound_valid_filt], s=12)
        ax.set_xlabel(r'$d^{\lambda}_{\mathrm{LLV}}$', fontsize = fontsize)
        ax.set_ylabel(r'$\max d_{\mathrm{SVD}}$', fontsize = fontsize)
        ax.legend(fontsize = fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        fig.tight_layout()
        figure_name = model_name + '_d_prob_vs_d_rep.png'
        figure_path = os.path.join(figure_folder, figure_name)
        plt.savefig(figure_path, dpi = dpi)

        # Plot t_1 vs min_m_SVDs 
        small_t_1_filt =  t_1s < 0.25

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(t_1s[small_t_1_filt], max_d_rep_g[small_t_1_filt], s=12)
        ax.plot(possible_distances, bound, label = 'bound')
        ax.set_xlabel('t_1', fontsize = fontsize)
        ax.set_ylabel(r'$\max d_{\mathrm{SVD}} \mathbf{g}$', fontsize = fontsize)
        ax.legend(fontsize = fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        fig.tight_layout()
        figure_name = model_name + '_t_1_vs_g_rep.png'
        figure_path = os.path.join(figure_folder, figure_name)
        plt.savefig(figure_path, dpi = dpi)
        #fig.show()

        # Plot t_2 vs min_m_SVDs 
        small_t_2_filt =  t_2s < 0.25

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(t_2s[small_t_2_filt], max_d_rep_f[small_t_2_filt], s=12)
        ax.plot(possible_distances, bound, label = 'bound')
        ax.set_xlabel('t_2', fontsize = fontsize)
        ax.set_ylabel(r'$\max d_{\mathrm{SVD}} \mathbf{f}$', fontsize = fontsize)
        ax.legend(fontsize = fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        fig.tight_layout()
        figure_name = model_name + '_t_2_vs_f_rep.png'
        figure_path = os.path.join(figure_folder, figure_name)
        plt.savefig(figure_path, dpi = dpi)
        #fig.show()
