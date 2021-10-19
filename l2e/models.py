"""
Collection of models for actor and critic
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym_physx.encoders import BaseEncoder


class RewardModel(nn.Module):
    """
    Reward prediction model
    """

    def __init__(
            self,
            achieved_goal_size,
            plan_size,
            plan_encoding_layers_dim,
            final_layers_dim
    ):
        super(RewardModel, self).__init__()

        self.plan_encoding_layers = nn.ModuleList()
        self.plan_encoding_layers.append(
            nn.Linear(
                plan_size,
                plan_encoding_layers_dim[0]
            )
        )
        for din, dout in zip(
                plan_encoding_layers_dim[:-1],
                plan_encoding_layers_dim[1:]
        ):
            self.plan_encoding_layers.append(
                nn.Linear(
                    din,
                    dout
                )
            )

        self.final_layers = nn.ModuleList()
        self.final_layers.append(
            nn.Linear(
                achieved_goal_size + plan_encoding_layers_dim[-1],
                final_layers_dim[0]
            )
        )
        for din, dout in zip(
                final_layers_dim[:-1],
                final_layers_dim[1:]
        ):
            self.final_layers.append(
                nn.Linear(
                    din,
                    dout
                )
            )
        self.final_layers.append(
            nn.Linear(
                final_layers_dim[-1],
                1
            )
        )

    def encode_plan(self, plan):
        """
        Plan encoding part of the network
        """
        for layer in self.plan_encoding_layers:
            plan = F.relu(layer(plan))

        return plan

    def forward(self, achieved_goal, plan):
        """
        Reward model forward pass
        """
        plan_encoding = self.encode_plan(plan)

        reward = torch.cat(
            [achieved_goal, plan_encoding],
            dim=-1
        )

        for layer in self.final_layers[:-1]:
            reward = F.relu(layer(reward))
        
        return self.final_layers[-1](reward)


class VanillaVAE(nn.Module):
    """
    VAE encoding for plans
    """

    def __init__(
            self,
            in_dim,
            latent_dim,
            intermediate_layers_dim,
            log_std_clips,
            device
    ):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        # create ModuleList() of linear layers for encoder
        self.e_linear_layers = nn.ModuleList()
        self.e_linear_layers.append(
            nn.Linear(
                in_dim,
                intermediate_layers_dim[0]
            )
        )
        for din, dout in zip(
                intermediate_layers_dim[:-1],
                intermediate_layers_dim[1:]
        ):
            self.e_linear_layers.append(
                nn.Linear(
                    din,
                    dout
                )
            )
        self.mean = nn.Linear(intermediate_layers_dim[-1], latent_dim)
        self.log_std = nn.Linear(intermediate_layers_dim[-1], latent_dim)

        # create ModuleList() of linear layers for decoder
        self.d_linear_layers = nn.ModuleList()
        self.d_linear_layers.append(
            nn.Linear(
                latent_dim,
                intermediate_layers_dim[-1]
            )
        )
        for din, dout in zip(
                intermediate_layers_dim[1:][::-1],
                intermediate_layers_dim[:-1][::-1]
        ):
            self.d_linear_layers.append(
                nn.Linear(
                    din,
                    dout
                )
            )
        self.d_linear_layers.append(
            nn.Linear(
                intermediate_layers_dim[0],
                in_dim
            )
        )

        self.latent_dim = latent_dim
        self.log_std_clips = log_std_clips
        self.device = device

    def forward(self, in_plan):
        """
        Forward pass through both encoder and decoder
        """
        mean, std = self.encode(in_plan)
        z_latent = mean + std * torch.randn_like(std)
        reconstruction = self.decode(z_latent=z_latent)

        return reconstruction

    def encode(self, in_plan):
        """
        Forward pass through encoder only
        """
        z_latent = in_plan

        for layer in self.e_linear_layers:
            z_latent = F.relu(layer(z_latent))

        mean = self.mean(z_latent)
        # Clamped for numerical stability
        log_std = self.log_std(z_latent).clamp(
            self.log_std_clips[0],
            self.log_std_clips[1]
        )
        std = torch.exp(log_std)

        return mean, std

    def decode(self, z_latent=None):
        """
        Forward pass through decoder only
        """
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z_latent is None:
            z_latent = torch.randn(self.latent_dim).to(
                self.device).clamp(-0.5, 0.5)

        reconstruction = z_latent
        for layer in self.d_linear_layers[:-1]:
            reconstruction = F.relu(layer(reconstruction))
        return self.d_linear_layers[-1](reconstruction)


class VaeEncoder(BaseEncoder):
    """
    Encoder class for VAE encoding
    """
    encoding_low = None
    encoding_high = None
    hidden_dim = None

    def __init__(self, vae: VanillaVAE, device: torch.device):
        self.vae = vae
        self.device = device

        # TODO this should not be hardcoded
        self.encoding_low = -15.
        self.encoding_high = 15.

        self.hidden_dim = self.vae.latent_dim # TODO confusing names

    def encode(self, plan: np.ndarray):
        """
        Convert to torch.Tensor, pass through vae.encode, convert back
        to np.ndarray
        """
        mean, _ = self.vae.encode(
            torch.Tensor(plan.reshape(1, -1)).to(self.device)
        )
        return mean.detach().numpy().reshape(-1)

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
    