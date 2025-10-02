""" Dataset config """



from dataclasses import dataclass




@dataclass
class DatasetConfig:
    """ Config class for datasets """
    random_seed: int
    
    dataset_name: str

    num_points: int 

    num_classes: int 

    equal_weighting_classes: bool

    min_length: float 

    dist_scale: float 
    vert_dist_scale: float 
