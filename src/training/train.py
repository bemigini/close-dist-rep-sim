"""

basic training loop


"""


import math

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.config.train_config import OptimizerType
from src.config.train_config import TrainConfig
from src.eval import eval_model 
from src.util import TargetType, convert_int_targets_to_one_hot



def train(
        train_dataloader: torch.Tensor, 
        test_dataloader: torch.Tensor,
        target_type: TargetType,
        model: nn.Module,
        train_config: TrainConfig,
        device: str,
        train_steps: int):
    """ For training a model """
    
    random_seed = train_config.random_seed
    torch.manual_seed(random_seed)
        
    learning_rate = train_config.learning_rate
    optimizer_type = train_config.optimizer_type    
    
    match target_type:
        case TargetType.CLASSIFICATION_BI:
            loss_fn = nn.BCELoss()
        
        case TargetType.CLASSIFICATION_MULTI:
            loss_fn = nn.CrossEntropyLoss()

        case TargetType.REGRESSION:
            loss_fn = nn.MSELoss()

        case _:
            raise ValueError(f"Unimplemented target type: {target_type.name}")
    
    
    match optimizer_type:
        case OptimizerType.SGD.name:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
            weight_decay = train_config.weight_decay)

        case OptimizerType.ADAM.name:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
            weight_decay = train_config.weight_decay)

        case _:
            raise ValueError(f"Unimplemented optimizer: {optimizer_type}")
    
    
    epochs = math.ceil(train_steps/len(train_dataloader))
    
    train_losses = []
    test_losses = []
    step = 0
    step_running_loss = 0.0
    model.to(device)
    
    for _ in tqdm(range(epochs)):  # loop over the dataset multiple times
        
        epoch_running_loss = 0.0
        for data in train_dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if len(labels.shape) == 1:
                labels = convert_int_targets_to_one_hot(labels, model.num_targets)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            if train_config.use_forward_as_loss:
                loss = model.forward(inputs, labels, device, train_config.reduction)            
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_running_loss += loss.item()
            step_running_loss += loss.item()
            
            # Add loss to train losses
            if step % 500 == 0:
                if step == 0:            
                    train_losses.append(step_running_loss)
                else:
                    train_losses.append(step_running_loss/500)
                step_running_loss = 0.0
            
            if step % 2000 == 0:
                eval_loss = eval_model(model, test_dataloader, loss_fn, 
                train_config.use_forward_as_loss, device)
                
                test_losses.append(eval_loss.detach().cpu().numpy())
                
            step += 1            
            
        # print every epoch
        print(f'loss: {epoch_running_loss:.3f}')       
            
            
    print('Finished Training')
    return train_losses, test_losses
    