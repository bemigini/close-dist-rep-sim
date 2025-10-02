"""

Dataset utils


"""


from enum import Enum

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn


class TargetType(Enum):
    """  Enum class for model target types """
    CLASSIFICATION_BI = 1
    CLASSIFICATION_MULTI = 2
    REGRESSION = 3


def batched_cos_sim(As: torch.Tensor, Bs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """ A batched cosine similarity where we expect As and Bs to be batches of vectors 
        of shape B x N.  
        eps is to avoid division by zero.
    """
    matrix_As = As.unsqueeze(1) # B x 1 X N
    matrix_Bs = Bs.unsqueeze(2) # B x N X 1

    dot_prod = torch.bmm(matrix_As, matrix_Bs) # B x 1 x 1
    dot_prod = dot_prod.squeeze(2).squeeze(1) # B
    cosine_sim = dot_prod/(torch.norm(As, dim = -1)*torch.norm(Bs, dim = -1) + eps)

    return cosine_sim


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """  For getting the cosine similarity between a and b """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            f'Cannot calculate cosine similarity for zero norm. norm_a:{norm_a}, norm_b:{norm_b}')

    cosine = np.dot(a,b)/(norm_a*norm_b)

    return cosine


def convert_int_targets_to_one_hot(int_targets, num_targets: int):
    """ Get one-hot encoded from ints """
    # pylint: disable=not-callable
    targets_oh = nn.functional.one_hot(
            int_targets, num_targets).float()
    targets_oh = targets_oh[:, :-1]
    
    return targets_oh


def convert_one_hot_to_ints(oh_targets):
    """ Get ints from one-hot encoded """
    num_targets = torch.tensor(oh_targets.shape[1])

    int_targets = []
    for current_oh_target in oh_targets:
        non_zero_idx = torch.nonzero(current_oh_target)
        if len(non_zero_idx) == 0:
            int_targets.append(num_targets)
        else:
            int_targets.append(non_zero_idx[0, 0])
    
    return int_targets
