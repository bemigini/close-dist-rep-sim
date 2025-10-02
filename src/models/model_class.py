"""

Model which fits the model class used in the article 


"""


import torch
import torch.nn as nn

from src.util import convert_int_targets_to_one_hot


class ModelClass(nn.Module):
    """ 
    Model made to fit the model class we consider in the article
    """
    def __init__(self, 
    num_targets:int,
    embedding_function: nn.Module,
    unembedding_function: nn.Module,
    fix_length_gs: float,
    fix_length_fs: float):
        super().__init__()
        
        self.num_targets = num_targets
        self.possible_targets = torch.arange(self.num_targets)
        self.possible_targets_oh = convert_int_targets_to_one_hot(
            self.possible_targets, self.num_targets)

        self.f_net = embedding_function
        self.g_net = unembedding_function

        self.eps = torch.tensor(1e-8)
        self.fix_length_gs = fix_length_gs
        self.fix_length_fs = fix_length_fs

    
    def dot(self, features, target):
        """ Dot product of net outputs """
        f_out = self.f_net(features)
        g_out = self.g_net(target)

        if self.fix_length_gs > 0:
            # pylint: disable=not-callable
            g_out = self.fix_length_gs * g_out / (
                torch.maximum(torch.linalg.norm(g_out, dim = -1, keepdim = True), self.eps))
        
        if self.fix_length_fs > 0:
            # pylint: disable=not-callable
            f_out = self.fix_length_fs * f_out / (
                torch.maximum(torch.linalg.norm(f_out, dim = -1, keepdim = True), self.eps))

        return torch.matmul(
                f_out.unsqueeze(1), g_out.unsqueeze(2))

    
    def log_likelihood_function(self, features, target, device) -> torch.Tensor:
        """ Calculating the log likelihood. Likelihood as in 
            Roeder et al. 2020 "On Linear Identifiability of Learned Representations"
            see equation 1. 
        """
            # p_{\theta}(y|x, S) = exp(f_{\theta}(x)^Tg_{\theta}(y)) / 
            #                        \sum_{y'\in S} (exp(f_{\theta}(x)^Tg_{\theta}(y')))
            # log(p) = f_{\theta}(x)^Tg_{\theta}(y) - 
            #                   log(\sum_{y'\in S} (exp(f_{\theta}(x)^Tg_{\theta}(y'))))
        
        val = self.dot(features, target).double()
        
        possible_targets_oh = self.possible_targets_oh.to(device) 
        normalisation = torch.zeros(
            (*val.shape, len(possible_targets_oh)), dtype=torch.float64).to(device)
        
        for i, current_target in enumerate(possible_targets_oh):
            n = self.dot(features, current_target.unsqueeze(0)).double()
            normalisation[:, :, :, i] = n 
        
        normalisation = torch.logsumexp(normalisation, dim = 3, keepdim = False)  
        return val - normalisation

    
    def forward(self, features, target, device, reduction = 'mean'):
        """ The forward pass. Getting the negative log-likelihood of the model. """        
        log_p = self.log_likelihood_function(features, target.float(), device)

        if reduction == 'mean':
            return -(log_p.mean())
        elif reduction == 'sum':
            return -(log_p.sum())
        else:
            raise ValueError(f'reduction should be either "mean" or "sum", reduction: {reduction} ')
        

    def predict_targets(self, features, device):
        """ Predicting the targets from the features """
        num_preds = features.shape[0]
        repeat_shape = torch.ones(len(features.shape)-1, dtype=int)

        use_features = features.unsqueeze(1).repeat((1, self.num_targets, *repeat_shape))
        use_features = torch.reshape(use_features, 
                                     (num_preds*self.num_targets, *features.shape[1:]))
        targets = self.possible_targets_oh.repeat((num_preds, 1))
        targets = targets.to(device)

        log_p = self.log_likelihood_function(use_features, targets, device)
        target_likelihoods = torch.reshape(log_p, (num_preds, self.num_targets))

        preds = torch.argmax(target_likelihoods, dim = -1)
        
        return preds


    def get_g_reps(self, target):
        """ Get model unembeddings """
        g_out = self.g_net(target)

        if self.fix_length_gs > 0:
            # pylint: disable=not-callable
            g_out = self.fix_length_gs * g_out / (
                torch.maximum(torch.linalg.norm(g_out, dim = -1, keepdim = True), self.eps))
        
        return g_out
    

    def get_f_reps(self, features):
        """ Get model embeddings """
        f_out = self.f_net(features)
        
        if self.fix_length_fs > 0:
            # pylint: disable=not-callable
            f_out = self.fix_length_fs * f_out / (
                torch.maximum(torch.linalg.norm(f_out, dim = -1, keepdim = True), self.eps))

        return f_out
    