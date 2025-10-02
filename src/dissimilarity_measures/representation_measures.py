""" 

Several ways of measuring dissimilarity between representations 


"""


from typing import List, Tuple

import numpy as np 
from numpy.typing import NDArray

from sklearn.cross_decomposition import CCA, PLSSVD
from sklearn.metrics import mean_squared_error

import torch
from tqdm import tqdm 




def get_all_possible_m_SVD(
        rep_dim: int,
        num_possible_pivots: int, 
        div_rep_indexes: torch.tensor,
        model1_div_reps: torch.tensor, model2_div_reps: torch.tensor, 
        model1_f_reps: torch.tensor, model2_f_reps: torch.tensor, 
        possible_diversity_combinations: List[Tuple]):
    """
        Get all the possible m_SVD of diversity matrices times representations

    """
    mean_svds_f = []                
    for c in tqdm(list(range(num_possible_pivots))):
        model1_div_reps_no_pivot = model1_div_reps[div_rep_indexes != c]
        model2_div_reps_no_pivot = model2_div_reps[div_rep_indexes != c]
        
        model1_0 = model1_div_reps[c]
        model2_0 = model2_div_reps[c]

        for current_comb in possible_diversity_combinations:
            model1_rest = model1_div_reps_no_pivot[list(current_comb)]
            L1_T = (model1_rest.double() - model1_0.double())

            model2_rest = model2_div_reps_no_pivot[list(current_comb)]
            L2_T = model2_rest.double() - model2_0.double()

            L1_T_f1 = torch.matmul(L1_T, model1_f_reps.double().unsqueeze(2)).squeeze() 
            L2_T_f2 = torch.matmul(L2_T, model2_f_reps.double().unsqueeze(2)).squeeze() 
            L1_T_f1_center = L1_T_f1 - torch.mean(L1_T_f1, dim=0)
            L2_T_f2_center = L2_T_f2 - torch.mean(L2_T_f2, dim=0)
            
            var_L1_T_f1 = torch.var(L1_T_f1_center, dim = 0, keepdims = True)
            var_L2_T_f2 = torch.var(L2_T_f2_center, dim = 0, keepdims = True)
            
            L1_T_f1_norm = L1_T_f1_center/torch.sqrt(var_L1_T_f1)
            L2_T_f2_norm = L2_T_f2_center/torch.sqrt(var_L2_T_f2)

            # Get PLS SVD mean covariance
            n_components = rep_dim
            plssvd = PLSSVD(n_components=n_components)
            plssvd.fit(L1_T_f1_norm, L2_T_f2_norm)
            X_c, Y_c = plssvd.transform(L1_T_f1_norm, L2_T_f2_norm)
            pls_svd_covs = [np.cov(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
            
            mean_svd_cov = np.mean(pls_svd_covs)
            mean_svds_f.append(mean_svd_cov)
    
    return mean_svds_f


def get_m_SVD_for_y_pivot(
        rep_dim: int,
        y_pivot_idx: int, 
        div_rep_indexes: torch.tensor,
        model1_div_reps: torch.tensor, model2_div_reps: torch.tensor, 
        model1_f_reps: torch.tensor, model2_f_reps: torch.tensor, 
        possible_diversity_combinations: List[Tuple]):
    """
        Get all the possible m_SVD of diversity matrices times representations

    """
    mean_svds_f = []                
    
    c = y_pivot_idx
    model1_div_reps_no_pivot = model1_div_reps[div_rep_indexes != c]
    model2_div_reps_no_pivot = model2_div_reps[div_rep_indexes != c]
    
    model1_0 = model1_div_reps[c]
    model2_0 = model2_div_reps[c]

    for current_comb in possible_diversity_combinations:
        model1_rest = model1_div_reps_no_pivot[list(current_comb)]
        L1_T = (model1_rest.double() - model1_0.double())

        model2_rest = model2_div_reps_no_pivot[list(current_comb)]
        L2_T = model2_rest.double() - model2_0.double()

        L1_T_f1 = torch.matmul(L1_T, model1_f_reps.double().unsqueeze(2)).squeeze() 
        L2_T_f2 = torch.matmul(L2_T, model2_f_reps.double().unsqueeze(2)).squeeze() 
        L1_T_f1_center = L1_T_f1 - torch.mean(L1_T_f1, dim=0)
        L2_T_f2_center = L2_T_f2 - torch.mean(L2_T_f2, dim=0)
        
        var_L1_T_f1 = torch.var(L1_T_f1_center, dim = 0, keepdims = True)
        var_L2_T_f2 = torch.var(L2_T_f2_center, dim = 0, keepdims = True)
        
        L1_T_f1_norm = L1_T_f1_center/torch.sqrt(var_L1_T_f1)
        L2_T_f2_norm = L2_T_f2_center/torch.sqrt(var_L2_T_f2)

        # Get PLS SVD mean covariance
        n_components = rep_dim
        plssvd = PLSSVD(n_components=n_components)
        plssvd.fit(L1_T_f1_norm, L2_T_f2_norm)
        X_c, Y_c = plssvd.transform(L1_T_f1_norm, L2_T_f2_norm)
        pls_svd_covs = [np.cov(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
        
        mean_svd_cov = np.mean(pls_svd_covs)
        mean_svds_f.append(mean_svd_cov)

    return mean_svds_f


def get_m_SVD_for_y_pivot_and_lo(
        rep_dim: int,
        y_pivot_idx: int, 
        lo_idx: int, 
        div_rep_indexes: torch.tensor,
        model1_div_reps: torch.tensor, model2_div_reps: torch.tensor, 
        model1_f_reps: torch.tensor, model2_f_reps: torch.tensor, 
        possible_diversity_combinations: List[Tuple]):
    """
        Get all the possible m_SVD of diversity matrices times representations

    """
    mean_svds_f = []                
    
    lo_filter = ~np.isin(div_rep_indexes, (y_pivot_idx, lo_idx))
    model1_div_reps_no_pivot_lo = model1_div_reps[lo_filter]
    model2_div_reps_no_pivot_lo = model2_div_reps[lo_filter]
    
    model1_0 = model1_div_reps[y_pivot_idx]
    model2_0 = model2_div_reps[y_pivot_idx]

    for current_comb in possible_diversity_combinations:
        model1_rest = model1_div_reps_no_pivot_lo[list(current_comb)]
        L1_T = (model1_rest.double() - model1_0.double())

        model2_rest = model2_div_reps_no_pivot_lo[list(current_comb)]
        L2_T = model2_rest.double() - model2_0.double()

        L1_T_f1 = torch.matmul(L1_T, model1_f_reps.double().unsqueeze(2)).squeeze() 
        L2_T_f2 = torch.matmul(L2_T, model2_f_reps.double().unsqueeze(2)).squeeze() 
        L1_T_f1_center = L1_T_f1 - torch.mean(L1_T_f1, dim=0)
        L2_T_f2_center = L2_T_f2 - torch.mean(L2_T_f2, dim=0)
        
        var_L1_T_f1 = torch.var(L1_T_f1_center, dim = 0, keepdims = True)
        var_L2_T_f2 = torch.var(L2_T_f2_center, dim = 0, keepdims = True)
        
        L1_T_f1_norm = L1_T_f1_center/torch.sqrt(var_L1_T_f1)
        L2_T_f2_norm = L2_T_f2_center/torch.sqrt(var_L2_T_f2)

        # Get PLS SVD mean covariance
        n_components = rep_dim
        plssvd = PLSSVD(n_components=n_components)
        plssvd.fit(L1_T_f1_norm, L2_T_f2_norm)
        X_c, Y_c = plssvd.transform(L1_T_f1_norm, L2_T_f2_norm)
        pls_svd_covs = [np.cov(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
        
        mean_svd_cov = np.mean(pls_svd_covs)
        mean_svds_f.append(mean_svd_cov)

    return mean_svds_f


def get_mSVD_chosen_input_set(
        div_input_idxs, m1_fx_reps, m2_fx_reps, model1_g_reps, model2_g_reps,
        num_classes, rep_dim):
    """ Get the idxs to use for diversity set """
    input_pivot_idx = div_input_idxs[0]
    div_input_rest = div_input_idxs[1:]

    f1_0 = m1_fx_reps[input_pivot_idx]
    f2_0 = m2_fx_reps[input_pivot_idx]

    f1_rest = m1_fx_reps[div_input_rest]
    f2_rest = m2_fx_reps[div_input_rest]

    N1_T = (f1_rest.double() - f1_0.double())        
    N2_T = f2_rest.double() - f2_0.double()
    # pylint: disable=not-callable 
    _, S_N1, _ = torch.linalg.svd(N1_T)
    _, S_N2, _ = torch.linalg.svd(N2_T)
    print(f'N1: {S_N1}')
    print(f'N2: {S_N2}')

    N1_T_g1 = torch.matmul(N1_T, model1_g_reps.double().unsqueeze(2)).squeeze() 
    N2_T_f2 = torch.matmul(N2_T, model2_g_reps.double().unsqueeze(2)).squeeze() 
    N1_T_g1_center = N1_T_g1 - torch.mean(N1_T_g1, dim=0)
    N2_T_g2_center = N2_T_f2 - torch.mean(N2_T_f2, dim=0)
    
    var_N1_T_g1 = torch.var(N1_T_g1_center, dim = 0, keepdims = True)
    var_N2_T_g2 = torch.var(N2_T_g2_center, dim = 0, keepdims = True)
    
    N1_T_g1_norm = N1_T_g1_center/torch.sqrt(var_N1_T_g1)
    N2_T_g2_norm = N2_T_g2_center/torch.sqrt(var_N2_T_g2)

    cross_cov_mat = torch.matmul(N1_T_g1_norm.T, N2_T_g2_norm)/num_classes
    _, S, _ = torch.linalg.svd(cross_cov_mat)
    print(f'cross cov: {S}')
    print()

    n_components = rep_dim
    plssvd = PLSSVD(n_components=n_components)
    plssvd.fit(N1_T_g1_norm, N2_T_g2_norm)
    X_c, Y_c = plssvd.transform(N1_T_g1_norm, N2_T_g2_norm)
    pls_svd_covs = [np.cov(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]

    mean_svd_cov_g = np.mean(pls_svd_covs)

    return mean_svd_cov_g


def get_sample_of_m_SVDs(
        rep_dim: int,
        num_samples_pivot: int,
        div_rep_indexes: torch.tensor,
        model1_div_reps: torch.tensor, model2_div_reps: torch.tensor, 
        model1_g_reps: torch.tensor, model2_g_reps: torch.tensor,
        alpha: float, use_alpha:bool,
        num_samples_rest: int = 10):
    """
        Get a sample of the possible m_SVD of diversity matrices times representations

    """
    mean_svds_g = []
    num_div_reps = len(div_rep_indexes)
    input_choices = np.random.choice(num_div_reps, num_samples_pivot, replace=False)
    for i, j in tqdm(enumerate(div_rep_indexes[input_choices])):
        f1_reps_no_pivot = model1_div_reps[div_rep_indexes != j]
        f2_reps_no_pivot = model2_div_reps[div_rep_indexes != j]
        
        f1_0 = model1_div_reps[j]
        f2_0 = model2_div_reps[j]

        # sample num_samples_rest options for other points
        idxs = np.random.choice(num_div_reps - 1, num_samples_rest*rep_dim, replace=False)
        idxs = np.reshape(idxs, (num_samples_rest, rep_dim))

        for current_idx in idxs:
            f1_rest = f1_reps_no_pivot[list(current_idx)]
            N1_T = (f1_rest.double() - f1_0.double())

            f2_rest = f2_reps_no_pivot[list(current_idx)]
            N2_T = f2_rest.double() - f2_0.double()

            # Don't use if either N1 or N2 is too close to not being invertible 
            # pylint: disable=not-callable 
            rank_N1 = torch.linalg.matrix_rank(N1_T)
            rank_N2 = torch.linalg.matrix_rank(N2_T)
            if (rank_N1 < rep_dim) or (rank_N2 < rep_dim):
                print(f'rank N1: {rank_N1}, rank N2: {rank_N2}')
                continue
            
            if use_alpha:
                _, S_N1, _ = torch.linalg.svd(N1_T)
                _, S_N2, _ = torch.linalg.svd(N2_T)
                if ((S_N1[-1]/S_N1.sum()) < alpha) or ((S_N2[-1]/S_N2.sum()) < alpha):
                    # print(S)
                    continue       

            N1_T_g1 = torch.matmul(N1_T, model1_g_reps.double().unsqueeze(2)).squeeze() 
            N2_T_f2 = torch.matmul(N2_T, model2_g_reps.double().unsqueeze(2)).squeeze() 
            N1_T_g1_center = N1_T_g1 - torch.mean(N1_T_g1, dim=0)
            N2_T_g2_center = N2_T_f2 - torch.mean(N2_T_f2, dim=0)
            
            var_N1_T_g1 = torch.var(N1_T_g1_center, dim = 0, keepdims = True)
            var_N2_T_g2 = torch.var(N2_T_g2_center, dim = 0, keepdims = True)
            
            N1_T_g1_norm = N1_T_g1_center/torch.sqrt(var_N1_T_g1)
            N2_T_g2_norm = N2_T_g2_center/torch.sqrt(var_N2_T_g2)

            # Check PLS SVD mean covariance
            n_components = rep_dim
            plssvd = PLSSVD(n_components=n_components)
            plssvd.fit(N1_T_g1_norm, N2_T_g2_norm)
            X_c, Y_c = plssvd.transform(N1_T_g1_norm, N2_T_g2_norm)
            pls_svd_covs = [np.cov(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]

            mean_svd_cov = np.mean(pls_svd_covs)
            mean_svds_g.append(mean_svd_cov)

        if i % 100 == 0:
            print(f'{i} min: {np.min(mean_svds_g)}')
    
    return mean_svds_g


def mCCA(rep1: NDArray, rep2: NDArray, n_components: int) -> float:
    """ Get mean canonical correlation of the two representations.
        Uses [n_components] components for the CCA.
    """
    cca = CCA(n_components=n_components, max_iter=1000)
    cca.fit(rep1, rep2)

    # Mean of the CCA correlations
    X_c, Y_c = cca.transform(rep1, rep2)
    corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
    mean_corr = np.mean(corrs)

    return mean_corr, cca.x_rotations_, cca.y_rotations_, np.max(corrs) 


def MSE(rep1: NDArray, rep2: NDArray) -> float:
    """ Mean squared error between representations 1 and 2 """
    return mean_squared_error(rep1, rep2)


def canonical_correlation_analysis(
    rep1: NDArray, rep2: NDArray, num_components: int) -> Tuple[NDArray, NDArray, float]:
    """ 
    Performing canonical correlation analysis on two representations using the specified 
    number of components. Returns the transformed representations and the CCA score.
    """
    cca = CCA(n_components=num_components)
    cca.fit(rep1, rep2)
    
    # Apply the transformation
    rep1_c, rep2_c = cca.transform(rep1, rep2)
    
    score = cca.score(rep1, rep2)

    return rep1_c, rep2_c, score


def left_KL_diag_measure(matrix: torch.Tensor):
    """ 
    As from Alyani 2017.     
    """
    symm_matrix = torch.matmul(matrix.T, matrix)
    diag_symm_m_05 = torch.diag(torch.diag(symm_matrix)**(-0.5))
    matrix_hat = torch.matmul(
        torch.matmul(diag_symm_m_05, symm_matrix), 
        diag_symm_m_05)
    # pylint: disable=not-callable
    sign, abs_det = torch.linalg.slogdet(matrix_hat)
    if sign < 0:
        raise ValueError('The symmetric matrix B^TB should be positive definite')
    return -1*abs_det
