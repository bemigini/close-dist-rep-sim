"""

Dataset like the one used to make figure 2 in 
Roeder 2020, "On Linear Identifiability of Learned Representations"


"""




import numpy as np

import torch
from torch.utils.data import Dataset

from src.util import TargetType


class RadialClassification(Dataset):
    """ Classification dataset with num_classes classes. Points are drawn from a 2 dimensional Gaussian, 
        N(0,3). Classes are radial. 
    """
    def __init__(self, 
                 num_points: int,
                 num_classes: int, 
                 random_seed: int,
                 equal_weighting_of_classes: bool = False,
                 eps: float = 1e-8):
        self.target_type = TargetType.CLASSIFICATION_MULTI
        
        self.num_points = num_points
        self.num_classes = num_classes
        self.random_seed = random_seed
        
        self.rng = np.random.default_rng(self.random_seed)
        self.data = torch.tensor(
            self.rng.normal(0, 3, (self.num_points, 2)), 
            dtype=torch.float32)
                        
        # \pi radians divided into num_classes
        num_radians = np.pi/num_classes
        possible_targets = np.arange(num_classes)
        self.possible_targets = possible_targets

        radian_bins = np.arange(0, np.pi, num_radians)
        cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
        targets = []
        unit_vec = torch.tensor([1., 0.])

        for i in range(self.data.shape[0]):
            current_point = np.copy(self.data[i])

            if current_point[1] < 0:
                current_point[1] = current_point[1] * -1
                current_point[0] = current_point[0] * -1

            current_cos = cos(torch.tensor(current_point), unit_vec)
            current_radian_val = np.arccos(current_cos)

            for j in possible_targets:
                current_bin = radian_bins[j]

                if j == (len(possible_targets) - 1):
                    targets.append(j)                    
                else:
                    if current_radian_val >= current_bin and current_radian_val < radian_bins[j+1]:
                        targets.append(j)
                        break
        
        self.targets = np.array(targets)
        
        targets_oh = np.zeros((self.targets.shape[0], self.num_classes - 1))
        for t in self.possible_targets:
            # Cut off last dimension of one-hot encoding to make it 
            # 17 dimensions for 18 classes
            # pylint: disable=not-callable
            targets_oh[self.targets == t] = torch.nn.functional.one_hot(
                torch.tensor(t, dtype = torch.int64), 
                self.num_classes).float().detach().numpy()[:-1]
        
        self.targets_oh = targets_oh

        
        if equal_weighting_of_classes:
            min_data = self.targets.shape[0]
            for t in self.possible_targets:
                num_data_in_class = np.sum(self.targets == t)
                if num_data_in_class < min_data:
                    min_data = num_data_in_class
            
            idxs_to_keep = np.zeros(min_data*self.num_classes, dtype = int)
            for i, t in enumerate(self.possible_targets):
                class_idxs = np.argwhere(self.targets == t)
                idxs_to_keep[i*min_data:(i + 1)*min_data] = class_idxs[:min_data, 0]
            
            self.rng.shuffle(idxs_to_keep)

            self.targets = self.targets[idxs_to_keep]
            self.data = self.data[idxs_to_keep]
            self.targets_oh = self.targets_oh[idxs_to_keep]        
        

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        point = self.data[idx]
        target = self.targets_oh[idx]
        
        return point, target
