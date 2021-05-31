"""
Collection of models for L2E
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseModel(nn.Module):
    """
    Model for learning the inverse dynamics
    """
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(InverseModel, self).__init__()

        in_dims = [2 * state_dim] + list(hidden_dims)
        out_dims = list(hidden_dims) + [action_dim]
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(in_dims, out_dims):
            self.layers.append(
                nn.Linear(in_dim, out_dim)
            )
    
    def forward(self, state, state_next):
        """
        Forward pass
        """
        tensor_in = torch.cat([state, state_next], axis=-1)

        for layer in self.layers[:-1]:
            tensor_in = F.relu(
                layer(tensor_in)
            )
        
        return self.layers[-1](tensor_in)
    