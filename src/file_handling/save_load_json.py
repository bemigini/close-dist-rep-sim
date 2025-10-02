"""

save and load json

"""


from typing import Callable

from enum import Enum

import json
from json import JSONEncoder

import numpy as np

import torch




class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.name
        if obj is None or isinstance(obj, Callable):
            return ''    
        return obj.__dict__


def save_as_json(obj: object, save_to: str) -> None:
    json_str = json.dumps(obj, indent = 4,
                          cls = CustomEncoder)
    
    with open(save_to, 'w') as f:
        f.write(json_str)


def load_json(load_from: str) -> str:
    
    with open(load_from, 'r') as f:
        json_obj = json.load(f)
        
    return json_obj
