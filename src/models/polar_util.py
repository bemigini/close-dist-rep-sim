"""

Utility functions for getting angles and cartesian coordinates with angles


"""


import numpy as np

import torch


def get_possible_targets_with_angles_2d(possible_targets: torch.Tensor):
    """ Get the possible targets with the angles in radians """
    Pi = torch.tensor(np.pi)
    num_classes = possible_targets.shape[0]
    g_angle = torch.zeros((num_classes, 1))
    for i in range(num_classes):
        g_angle[i, 0] = i*2*Pi/num_classes
    return g_angle
    

def get_fixed_g_angle_2d(
    target: torch.Tensor, possible_targets: torch.Tensor, fixed_gs: torch.Tensor):
    """ Get the g angles when gs are fixed """
    num_classes = fixed_gs.shape[0]
    g_angle = torch.zeros((target.shape[0], 1))
    for i in range(num_classes):
        g_angle[torch.all(target == possible_targets[i], dim = 1)] = fixed_gs[i]
    
    return g_angle


def get_cartesian_coordinates_from_radians(radians):
    """ Get cartesian coordinates from angles in radians """
    # "Polar" coordinates, M dimension of representations space
    # For angles \phi_i, i \in {1, ..., M - 1}
    # For vector values v_j, j \in {0, ..., M - 1}
    # We let length of vectors be equal to 1
    # TODO: Use Euler's formula to avoid products?
    M_dim = radians.shape[1] + 1
    vectors = torch.zeros((radians.shape[0], M_dim)) # B x M
    # First coordinate comes from only cosine
    vectors[:, 0] = torch.prod(torch.cos(radians), dim = -1) # B
    # Last coordinate comes from only sine
    vectors[:, -1] = torch.sin(radians[:, -1]) # B
    # Coordinates in between are mixed 
    for j in range(1, M_dim-1):
        cos_part = torch.prod(torch.cos(radians[:, j:]), dim = -1) # B
        sin_part = torch.sin(radians[:, j-1]) # B
        vectors[:, j] = sin_part * cos_part # B
    
    return vectors


