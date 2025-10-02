
"""

Model config


"""



from dataclasses import dataclass

from typing import List 

from src.util import TargetType



@dataclass
class ModelConfig:
    """ Config class for a model """
    random_seed: int
    
    model_type: str # SmallMLP

    target_type: TargetType
    nonlinearity: str

    num_classes: int 
    num_features: int
    rep_dim: int
    fix_length_gs: float
    fix_length_fs: float


@dataclass
class ModelVariationsConfig:
    """ Config class for a number of models """
    random_seeds: List[int]
    
    model_type: str # SmallMLP

    target_type: TargetType
    nonlinearity: str

    num_classes: int
    num_features: List[int]
    rep_dim: int
    fix_length_gs: List[float]
    fix_length_fs: List[float]
