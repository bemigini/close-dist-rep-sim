"""

Evaluate model


"""


import torch

from torch.utils.data import DataLoader

from src.models import model_class
from src.util import convert_int_targets_to_one_hot


def eval_model(
    model: model_class.ModelClass, test_dataloader: DataLoader, loss_fn,
    use_forward_as_loss: bool, device: str):
    """ Evaluate a model """
    running_test_loss = 0
    for test_data in test_dataloader:
        inputs, labels = test_data

        if len(labels.shape) == 1:
            labels = convert_int_targets_to_one_hot(labels, model.num_targets)

        inputs = inputs.to(device)
        labels = labels.to(device)

        if use_forward_as_loss:
            with torch.no_grad():
                loss = model(inputs, labels, device)
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
        running_test_loss += loss

    final_loss = running_test_loss/len(test_dataloader)
    return final_loss
