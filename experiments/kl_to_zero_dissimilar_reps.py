"""

Implementation of example where we can make KL-divergence go to zero, 
but where representations are dissimilar. 


"""


import itertools

import numpy as np 

import pandas as pd

from sklearn.cross_decomposition import CCA

import torch 

from experiments.making_constructed_models import make_even_spaced_classifier_2d
from experiments.making_constructed_models import get_gy_from_rad_and_length, get_fx_from_rad_from_g_and_length

import src.dissimilarity_measures.distribution_distance as dd
import src.dissimilarity_measures.representation_measures as rep_m



def make_pair_of_small_KL_models(
        g_length: float, num_classes: int, min_f_length: float):
    """ Make a pair of models with small KL-divergence, but dissimilar representations """    
    num_inputs_per_class = 500

    m1_all_fx, m1_gy, g_m1_angles, x_angles_m1, fx_lengths = make_even_spaced_classifier_2d(
            num_classes=num_classes, num_inputs_per_class=num_inputs_per_class, 
            g_length=g_length, min_f_length=min_f_length)
    
    # Model 2 is model 1 with labels permuted 
    model1_to_2_permutation = np.array([5, 0, 2, 4, 6, 1, 3])

    g_m2_angles = g_m1_angles[model1_to_2_permutation]
    x_angles_m2 = x_angles_m1[model1_to_2_permutation] 

    m2_gy = get_gy_from_rad_and_length(g_m2_angles, g_length)

    m2_fx = get_fx_from_rad_from_g_and_length(x_angles_m2, g_m2_angles, fx_lengths)
    m2_all_fx = np.concatenate(m2_fx, axis = 0)

    return m1_all_fx, m1_gy, m2_all_fx, m2_gy


def kl_to_zero_dissimilar_reps(device: str):
    """ 
        Let KL-divergence go to zero by increasing g length
        See that d_prob and representational similarity does not go down.      
    """
    num_classes = 7
    weight = 0.00001
    num_samples=100
    dist_type='max'
    sum_or_max = 'max'
    g_lengths = np.arange(1, 21)
    min_f_length = 2

    rep_dim = 2
    g_indexes = np.arange(num_classes)

    possible_diversity_targets_g = range(num_classes-2)
    possible_diversity_combinations_g = list(
        itertools.combinations(possible_diversity_targets_g, rep_dim))
    
    table_dict = {
        'g_length': [],
        'KL_div': [],
        'd_prob': [],
        'mCCA_f': [],
        'mCCA_g': [],
        'max_d_rep_f': [],
        'max_d_rep_g': []
    }

    for current_length in g_lengths:
        m1_all_fx, m1_gy, m2_all_fx, m2_gy = make_pair_of_small_KL_models(
            current_length, num_classes, min_f_length)

        f1_reps = torch.from_numpy(m1_all_fx)
        g1_reps = torch.from_numpy(m1_gy).squeeze()
        f2_reps = torch.from_numpy(m2_all_fx)
        g2_reps = torch.from_numpy(m2_gy).squeeze()

        # The KL-divergence
        current_kl_div = dd.get_mean_KL_divergence(
            f1_reps, g1_reps, f2_reps, g2_reps, device=device
        )
        print(f'{current_length} KL: {current_kl_div}')

        # The d_prob distance
        d_prob, _, _, input_idxs, y_pivot_idx, lo_idx  = dd.d_prob(
            f1_reps, g1_reps, f2_reps, g2_reps, 
            num_classes, weight=weight, num_samples=num_samples, 
            device=device, dist_type=dist_type, sum_or_max=sum_or_max, 
            use_div_idxs=None, use_y_pivot_idx=-1,
        )
        print(f'{current_length} d_prob: {d_prob}')

        # mCCA
        cca = CCA(n_components=rep_dim, max_iter=1000)
        cca.fit(f1_reps, f2_reps)
        X_c, Y_c = cca.transform(f1_reps, f2_reps)
        corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(rep_dim)]
        print(f'{current_length} mCCA f correlations: {corrs}')
        mean_corr_f = np.mean(corrs)
        print(f'{current_length} mCCA f: {mean_corr_f}')

        try:
            cca = CCA(n_components=rep_dim, max_iter=1000)
            cca.fit(g1_reps, g2_reps)
            X_c, Y_c = cca.transform(g1_reps, g2_reps)
            corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(rep_dim)]
            print(f'{current_length} mCCA g correlations: {corrs}')
            mean_corr_g = np.mean(corrs)
            print(f'{current_length} mCCA g: {mean_corr_g}')
        # pylint: disable=bare-except
        except:
            mean_corr_g = np.nan
            print(f'{current_length} mCCA g: {mean_corr_g}')

        
        # mSVD
        try:
            mean_svds_f = rep_m.get_m_SVD_for_y_pivot_and_lo(
                rep_dim=rep_dim, y_pivot_idx=y_pivot_idx.numpy(), 
                lo_idx=lo_idx.numpy(),
                div_rep_indexes=g_indexes, 
                model1_div_reps=g1_reps, model2_div_reps=g2_reps,
                model1_f_reps=f1_reps, model2_f_reps=f2_reps, 
                possible_diversity_combinations=possible_diversity_combinations_g
                )
            print(f'{current_length} max d_rep f: {1-np.min(mean_svds_f)}')
        # pylint: disable=bare-except
        except:
            mean_svds_f = np.nan
            print(f'{current_length} max d_rep f: nan')
        
        mean_svd_cov_g = rep_m.get_mSVD_chosen_input_set(
            input_idxs, f1_reps, f2_reps, g1_reps, g2_reps, 
            num_classes=num_classes, rep_dim=rep_dim
        )
        mean_svds_g = [mean_svd_cov_g]
        print(f'{current_length} mSVD gs: {mean_svds_g}')

        table_dict['g_length'].append(current_length)
        table_dict['KL_div'].append(float(current_kl_div))
        table_dict['d_prob'].append(d_prob)
        table_dict['mCCA_f'].append(mean_corr_f)
        table_dict['mCCA_g'].append(mean_corr_g)
        table_dict['max_d_rep_f'].append(np.max(1-np.array(mean_svds_f)))
        table_dict['max_d_rep_g'].append(np.max(1-np.array(mean_svds_g)))

    return table_dict


def make_latex_table_from_dict(table_dict):
    """ Make a latex table from a dictionary """
    table_dict.pop('mCCA_g', None)
    table_dict.pop('max_d_rep_g', None)

    df = pd.DataFrame.from_dict(table_dict)
    latex = df.to_latex(index=False, float_format="{:.4f}".format)
    latex = latex.replace('_', ' ')
    latex = latex.replace(r'\toprule', r'\hline')
    latex = latex.replace(r'\midrule', r'\hline')
    latex = latex.replace(r'\bottomrule', r'\hline')

    return latex
