"""


For making the constructed models 



"""


import itertools


import numpy as np 
from numpy.typing import NDArray

import torch

from src.dissimilarity_measures.distribution_distance import d_prob
from src.dissimilarity_measures.representation_measures import get_m_SVD_for_y_pivot_and_lo, get_mSVD_chosen_input_set


def get_fx_from_rad_from_g_and_length(
    f_rad_from_g: NDArray, g_angle: NDArray, f_length: NDArray):
    """
    Get the embedding functions based on the angles of the unembedding vectors 
    and the radian from the correct unembedding 
    """
    f_angle = (g_angle + f_rad_from_g) % (2*np.pi)
    fx_a = f_length*np.cos(f_angle)
    fx_b = f_length*np.sin(f_angle)
    fx = np.concatenate((np.expand_dims(fx_a, 2), np.expand_dims(fx_b, 2)), 2)

    return fx 


def get_gy_from_rad_and_length(
    g_rad: NDArray, g_length: float):
    """ 
    Get the unembedding vectors from the angle in radians
    and the length of the vectors. 
    """
    gy_a = g_length*np.cos(g_rad)
    gy_b = g_length*np.sin(g_rad)
    gy = np.concatenate((np.expand_dims(gy_a, 2), np.expand_dims(gy_b, 2)), 2)

    return gy


def make_even_spaced_classifier_2d(
        num_classes: int, num_inputs_per_class: int, 
        min_f_length: float = 5, g_length: float = 20):
    """ Make a classification model with evenly spaced unembeddings """
    rng = np.random.default_rng()
    num_inputs = num_inputs_per_class * num_classes

    even_spaced_angles = [i * (2*np.pi/num_classes) for i in range(num_classes)]
    g_m1_angles = np.expand_dims(np.array(even_spaced_angles), 1)
    x_angles_m1 = rng.uniform(-np.pi/64, np.pi/64, (num_classes, int(num_inputs/num_classes)))
    fx_lengths = np.abs(
        0.01*rng.standard_normal(
            (num_classes, int(num_inputs/num_classes)))) + min_f_length

    m1_gy = get_gy_from_rad_and_length(g_m1_angles, g_length)

    m1_fx = get_fx_from_rad_from_g_and_length(x_angles_m1, g_m1_angles, fx_lengths)
    m1_all_fx = np.concatenate(m1_fx, axis = 0)

    return m1_all_fx, m1_gy, g_m1_angles, x_angles_m1, fx_lengths


def get_distances_to_perturbed_models(
        m1_all_fx, m1_gy, g_m1_angles, x_angles_m1, fx_lengths, 
        num_classes: int, noise_level: float):
    """ Make perturbed models and get distances """
    rng = np.random.default_rng()
    rep_dim = 2
    weight = 0.00001
    num_samples = 300
    g_indexes = np.arange(num_classes)

    possible_diversity_targets_g = range(num_classes-2)
    possible_diversity_combinations_g = list(
        itertools.combinations(possible_diversity_targets_g, rep_dim))

    m1_fx_reps = torch.from_numpy(m1_all_fx)
    m1_gy_reps = torch.from_numpy(np.squeeze(m1_gy))

    dist_type = 'max'
    dist_bound_dict = {
        'd_prob':[], 'd_rep_bound': [], 't_1': [], 't_2': [], 'msvds_Lfx': [], 'msvds_Ngy': []}
    div_input_idx_choices = None
    y_pivot = -1
    lo_idx = -1
    for i in range(250):
        g_m2_angles = g_m1_angles

        m2_gy = get_gy_from_rad_and_length(g_m2_angles, 20)


        m2_fx = get_fx_from_rad_from_g_and_length(x_angles_m1, g_m2_angles, fx_lengths)
        m2_all_fx = np.concatenate(m2_fx, axis = 0)
        small_noise = (i+1)*noise_level*rng.standard_normal(m2_all_fx.shape)
        m2_all_fx = m2_all_fx+small_noise
        m2_fx_reps = torch.from_numpy(m2_all_fx)       
        m2_gy_reps = torch.from_numpy(np.squeeze(m2_gy))

        m1_m2_distance, t_1, t_2, div_input_idx_choices, y_pivot_idx, lo_idx = d_prob(
            m1_fx_reps, m1_gy_reps, 
            m2_fx_reps, m2_gy_reps, 
            num_classes, weight, num_samples, 
            device='cpu', dist_type=dist_type, sum_or_max='max', 
            use_div_idxs=div_input_idx_choices, use_y_pivot_idx=y_pivot)
        m1_m2_bound = 2*rep_dim*m1_m2_distance
        print(f'{i}:{m1_m2_distance}')
        
        mean_svds_f = get_m_SVD_for_y_pivot_and_lo(
            rep_dim=rep_dim, y_pivot_idx=y_pivot_idx.numpy(), 
            lo_idx=lo_idx,
            div_rep_indexes=g_indexes, 
            model1_div_reps=m1_gy_reps, model2_div_reps=m2_gy_reps,
            model1_f_reps=m1_fx_reps, model2_f_reps=m2_fx_reps, 
            possible_diversity_combinations=possible_diversity_combinations_g
            )
        mean_svd_cov_g = get_mSVD_chosen_input_set(
            div_input_idx_choices, 
            m1_fx_reps, m2_fx_reps, m1_gy_reps, m2_gy_reps, 
            num_classes=num_classes, rep_dim=rep_dim
        )
        mean_svds_g = [mean_svd_cov_g]        

        dist_bound_dict['d_prob'].append(m1_m2_distance)
        dist_bound_dict['d_rep_bound'].append(m1_m2_bound)
        dist_bound_dict['t_1'].append(t_1)
        dist_bound_dict['t_2'].append(t_2)
        dist_bound_dict['msvds_Lfx'].append(mean_svds_f)
        dist_bound_dict['msvds_Ngy'].append(mean_svds_g)

    return dist_bound_dict
