"""
Training config 

"""

from dataclasses import dataclass

from enum import Enum


class OptimizerType(Enum):
    """ Optimizer enum """
    SGD = 1
    ADAM = 2

@dataclass
class TrainConfig:
    """ Data class for holding training configuration """
    dataset_name: str
    random_seed: int
    
    batch_size: int
    train_steps: int
    
    reduction: str
    use_forward_as_loss: bool
    
    learning_rate: float
    optimizer_type: OptimizerType
    weight_decay: float = 0 
