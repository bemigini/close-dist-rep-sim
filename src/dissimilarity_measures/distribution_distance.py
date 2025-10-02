"""


Implementation of the distance between sets of probability distributions, 
d_prob, for models of the model class. 



"""


import numpy as np

from tqdm import tqdm
from typing import Tuple

import torch



def get_terms_1_2(ln_p1_y_x, ln_p1_y0_x, ln_p2_y_x, ln_p2_y0_x, dim:int) -> float:
    """
    ln_p1_y_x: the log likelihoods of y given x under the first model
    ln_p1_y0_x: the log likelihoods of y_0, the "pivot point", given x under the first model
    ln_p2_y_x: the log likelihoods of y given under the second model
    ln_p2_y0_x: the log likelihoods of y_0, the "pivot point", given x under the second model

    returns: term_vals to use for calculating distance term
    """

    var_x_ln_diff_p1 = torch.var(ln_p1_y_x - ln_p1_y0_x, dim=dim, keepdim=True)    
    var_x_ln_diff_p2 = torch.var(ln_p2_y_x - ln_p2_y0_x, dim=dim, keepdim=True)

    terms = torch.sqrt(
        torch.var(
            (ln_p1_y_x/torch.sqrt(var_x_ln_diff_p1)) - (ln_p2_y_x/torch.sqrt(var_x_ln_diff_p2)), 
            dim = dim))

    return terms


def get_terms_3_4(ln_p1_y_x, ln_p1_y0_x, ln_p2_y_x, ln_p2_y0_x, dim:int) -> float:
    """
    ln_p1_y_x: the log likelihoods of y given x under the first model
    ln_p1_y0_x: the log likelihoods of y_0, the "pivot point", given x under the first model
    ln_p2_y_x: the log likelihoods of y given x under the second model
    ln_p2_y0_x: the log likelihoods of y_0, the "pivot point", given x under the second model

    returns: term_vals to use for calculating distance term
    """

    var_x_ln_diff_p1 = torch.var(ln_p1_y_x - ln_p1_y0_x, dim = dim)
    var_x_ln_diff_p2 = torch.var(ln_p2_y_x - ln_p2_y0_x, dim = dim)

    terms = torch.abs(torch.sqrt(var_x_ln_diff_p1) - torch.sqrt(var_x_ln_diff_p2))

    return terms


def get_terms_1_2_pivot(ln_p1_y_x, ln_p1_y0_x, ln_p2_y_x, ln_p2_y0_x, dim:int) -> torch.tensor:
    """
    ln_p1_y_x: the log likelihoods of y given x under the first model
    ln_p1_y0_x: the log likelihoods of y_0, the "pivot point", given x under the first model
    ln_p2_y_x: the log likelihoods of y given under the second model
    ln_p2_y0_x: the log likelihoods of y_0, the "pivot point", given x under the second model

    returns: term_vals to use for calculating distance term
    """

    var_x_ln_diff_p1 = torch.var(ln_p1_y_x - ln_p1_y0_x, dim=dim, keepdim=True)    
    var_x_ln_diff_p2 = torch.var(ln_p2_y_x - ln_p2_y0_x, dim=dim, keepdim=True)

    terms_rest = torch.sqrt(
        torch.var(
            (ln_p1_y_x/torch.sqrt(var_x_ln_diff_p1)) - (ln_p2_y_x/torch.sqrt(var_x_ln_diff_p2)), 
            dim = dim))

    terms_pivot = torch.sqrt(
        torch.var(
            (ln_p1_y0_x/torch.sqrt(var_x_ln_diff_p1)) - (ln_p2_y0_x/torch.sqrt(var_x_ln_diff_p2)), 
            dim = dim))
    
    terms = torch.cat([terms_rest, terms_pivot], 0)
    
    return terms


def get_input_set_idxs_t2(
        ln_p1_y_x: torch.tensor, ln_p2_y_x: torch.tensor, rep_dim: int,
        num_samples: int):
    """ Get indexes of inputs to use for diversity set """
    num_inputs = ln_p1_y_x.shape[1]

    input_choices = np.random.choice(num_inputs, (rep_dim + 1)*num_samples, replace=False)
    input_choices = np.reshape(input_choices, (num_samples, rep_dim + 1))

    min_max_t2 = 1
    min_t2_choice = input_choices[0]
    for current_choice in input_choices:
        input_pivot_idx = current_choice[0]
        div_input_rest = current_choice[1:]

        ln_p1_y_x0 = torch.unsqueeze(ln_p1_y_x[:, input_pivot_idx], 1)
        ln_p2_y_x0 = torch.unsqueeze(ln_p2_y_x[:, input_pivot_idx], 1)

        ln_p1_y_rest_x = ln_p1_y_x[:, div_input_rest]
        ln_p2_y_rest_x = ln_p2_y_x[:, div_input_rest]

        term_2 = get_terms_1_2_pivot(ln_p1_y_rest_x, ln_p1_y_x0, ln_p2_y_rest_x, ln_p2_y_x0, 0)
        max_term_2 = torch.max(term_2)
        if max_term_2 < min_max_t2:
            min_max_t2 = max_term_2
            min_t2_choice = current_choice

    return min_t2_choice, min_max_t2


def choose_y_pivot(
        ln_p1_yks_x, ln_p2_yks_x, 
        num_classes:int, dist_type: str):
    """ Get the best y pivot to use """
    terms_1 = torch.zeros(num_classes)

    for k in tqdm(np.arange(num_classes)):
        term_1, _ = get_term_1_and_3_for_unembeddings(
        ln_p1_yks_x, ln_p2_yks_x, num_classes, k
        )

        if dist_type == 'max':
            terms_1[k] = torch.max(term_1)
        elif dist_type == 'mean':
            terms_1[k] = torch.mean(term_1)
        else:
            raise ValueError(f'Unknown distance type: {dist_type}')
    
    min_idx = torch.argmin(terms_1)

    return min_idx


def choose_y_pivot_and_leave_out(
        ln_p1_yks_x, ln_p2_yks_x, 
        num_classes:int, dist_type: str):
    """ Get the best y pivot to use """
    terms_1 = torch.zeros((num_classes, num_classes - 1))
    leave_out_possibilities = np.arange(num_classes-1)
    for_leave_out_filter = np.arange((num_classes-1)*2)

    for k in tqdm(np.arange(num_classes)):
        term_1, _ = get_term_1_and_3_for_unembeddings(
        ln_p1_yks_x, ln_p2_yks_x, num_classes, k
        )

        for l in leave_out_possibilities:
            lo_filter = ~np.isin(
                for_leave_out_filter, (l, l+num_classes-1))

            if dist_type == 'max':
                terms_1[k, l] = torch.max(term_1[lo_filter])
            elif dist_type == 'mean':
                terms_1[k, l] = torch.mean(term_1[lo_filter])
            else:
                raise ValueError(f'Unknown distance type: {dist_type}')

    min_pivot_idx, lo_idx = (terms_1==torch.min(terms_1)).nonzero()[0]
    if lo_idx >= min_pivot_idx:
        lo_idx = lo_idx + 1

    return min_pivot_idx, lo_idx


def get_term_1_and_3_for_unembeddings(
        ln_p1_yks_x, ln_p2_yks_x, num_classes:int, pivot_idx: int):
    """ Get term 1 and 3 for unembeddings """
    ln_p1_y0_x = torch.unsqueeze(ln_p1_yks_x[pivot_idx], 0)
    ln_p2_y0_x = torch.unsqueeze(ln_p2_yks_x[pivot_idx], 0)

    pivot_filter = torch.arange(num_classes) != pivot_idx
    ln_p1_y_rest_x = ln_p1_yks_x[pivot_filter]
    ln_p2_y_rest_x = ln_p2_yks_x[pivot_filter]

    # var over x
    term_1 = get_terms_1_2_pivot(ln_p1_y_rest_x, ln_p1_y0_x, ln_p2_y_rest_x, ln_p2_y0_x, 1)
    term_3 = get_terms_3_4(ln_p1_y_rest_x, ln_p1_y0_x, ln_p2_y_rest_x, ln_p2_y0_x, 1)

    return term_1, term_3


def log_likelihood_from_reps(features, targets) -> torch.Tensor:
    """ Calculating the log likelihood. 
    """
        # p_{\theta}(y|x, S) = exp(f_{\theta}(x)^Tg_{\theta}(y)) / 
        #                        \sum_{y'\in S} (exp(f_{\theta}(x)^Tg_{\theta}(y')))
        # log(p) = f_{\theta}(x)^Tg_{\theta}(y) - 
        #                   log(\sum_{y'\in S} (exp(f_{\theta}(x)^Tg_{\theta}(y'))))
    num_classes = targets.shape[0]
    vals = torch.zeros((num_classes, features.shape[0]))
    for k, _ in enumerate(targets):
        g1_0 = torch.unsqueeze(targets[k], 0)
        current_val = torch.matmul(features, g1_0.T)
        vals[k] = current_val.squeeze()
        
    normalisation = torch.logsumexp(vals, dim = 0, keepdim = True)  
    return vals - normalisation


def d_prob(
        f1_reps, g1_reps, f2_reps, g2_reps, num_classes: int,
        weight: float, num_samples: int, 
        device: str, dist_type:str = 'max', sum_or_max:str = 'sum',
        use_div_idxs = None, use_y_pivot_idx: int = -1,
        use_lo_idx: int = -1) -> Tuple[float, float, float, int, int]:
    """
    f_reps: list of tensors, each tensor is the embedding representation of an input.
    g_reps: list of tensors, each tensor is the unembedding representation of a label.
    num_classes: number of classes for the model
    rep_dim: dimensionality of the model representations

    returns: distance
    
    """
    f1_reps = f1_reps.to(device)  
    g1_reps = g1_reps.to(device)
    f2_reps = f2_reps.to(device)
    g2_reps = g2_reps.to(device)

    ln_p1_yks_x = log_likelihood_from_reps(f1_reps, g1_reps)
    ln_p2_yks_x = log_likelihood_from_reps(f2_reps, g2_reps)
    
    # terms only have the element from the input set and chosen label pivot 
    terms_1 = torch.zeros(1)
    terms_3 = torch.zeros(1)
    terms_2 = torch.zeros(1)
    terms_4 = torch.zeros(1)


    if use_y_pivot_idx > -1:
        y_pivot_idx = use_y_pivot_idx
        lo_idx = use_lo_idx
    else:
        y_pivot_idx, lo_idx = choose_y_pivot_and_leave_out(
            ln_p1_yks_x, ln_p2_yks_x, num_classes, dist_type)
    
    lo_filter = torch.arange(num_classes) != lo_idx

    ln_p1_t13 = ln_p1_yks_x[lo_filter]
    ln_p2_t13 = ln_p2_yks_x[lo_filter]
    if lo_idx < y_pivot_idx:
        current_y_pivot_idx = y_pivot_idx - 1
    else:
        current_y_pivot_idx = y_pivot_idx
    
    term_1, term_3 = get_term_1_and_3_for_unembeddings(
            ln_p1_t13, ln_p2_t13, num_classes-1, current_y_pivot_idx
        )

    if dist_type == 'max':
        terms_1[0] = torch.max(term_1)
        terms_3[0] = torch.max(term_3)
    elif dist_type == 'mean':
        terms_1[0] = torch.mean(term_1)
        terms_3[0] = torch.mean(term_3)
    else:
        raise ValueError(f'Unknown distance type: {dist_type}')
    t_1 = torch.max(terms_1)
    print(f't1: {t_1}')

    if (use_div_idxs is not None) and (len(use_div_idxs) > 0):
        div_input_idxs = use_div_idxs
    else:
        rep_dim = f1_reps.shape[1]
        div_input_idxs, _ = get_input_set_idxs_t2(
            ln_p1_yks_x, ln_p2_yks_x, rep_dim, num_samples=num_samples)
        
    
    input_pivot_idx = div_input_idxs[0]
    div_input_rest = div_input_idxs[1:]

    ln_p1_y_x0 = torch.unsqueeze(ln_p1_yks_x[:, input_pivot_idx], 1)
    ln_p2_y_x0 = torch.unsqueeze(ln_p2_yks_x[:, input_pivot_idx], 1)

    ln_p1_y_rest_x = ln_p1_yks_x[:, div_input_rest]
    ln_p2_y_rest_x = ln_p2_yks_x[:, div_input_rest]

    term_2 = get_terms_1_2_pivot(ln_p1_y_rest_x, ln_p1_y_x0, ln_p2_y_rest_x, ln_p2_y_x0, 0)
    term_4 = get_terms_3_4(ln_p1_y_rest_x, ln_p1_y_x0, ln_p2_y_rest_x, ln_p2_y_x0, 0)

    if dist_type == 'max':
        terms_2[0] = torch.max(term_2)
        terms_4[0] = torch.max(term_4)
    elif dist_type == 'mean':
        # Always take max of term 2
        terms_2[0] = torch.max(term_2)
        terms_4[0] = torch.mean(term_4)
    else:
        raise ValueError(f'Unknown distance type: {dist_type}')
    t_2 = terms_2[0]
    print(f't2: {t_2}')
    
    if sum_or_max == 'max':
        distance = np.max((torch.max(terms_1), torch.max(terms_2), 
                weight*torch.max(terms_3), weight*torch.max(terms_4)))
    else:
        distance = (torch.max(terms_1) + torch.max(terms_2) 
                + weight*torch.max(terms_3) + weight*torch.max(terms_4))

    return distance, t_1, t_2, div_input_idxs, y_pivot_idx, lo_idx


def get_mean_KL_divergence(
        f1_reps, g1_reps, f2_reps, g2_reps, 
        device: str):
    """ Get the average KL-divergence of two models """
    f1_reps = f1_reps.to(device)  
    g1_reps = g1_reps.to(device)
    f2_reps = f2_reps.to(device)
    g2_reps = g2_reps.to(device)
    
    num_inputs = f1_reps.shape[0]
    
    ln_p1_yks_x = log_likelihood_from_reps(f1_reps, g1_reps)
    ln_p2_yks_x = log_likelihood_from_reps(f2_reps, g2_reps)
    
    p1_probs = torch.exp(ln_p1_yks_x)
    
    kl_divergences = torch.zeros(num_inputs)
    for i in range(num_inputs):
        ln_p1_y_xi = ln_p1_yks_x[:, i]
        ln_p2_y_xi = ln_p2_yks_x[:, i] 

        current_p1_probs = p1_probs[:, i]
        current_kl_12_div = (current_p1_probs * (ln_p1_y_xi - ln_p2_y_xi)).sum()
        kl_divergences[i] = current_kl_12_div
    
    mean_kl_div = kl_divergences.sum()/num_inputs

    return mean_kl_div
