"""Neural network model for backward diffusion - noise prediction epsilon_theta(x_t, t)."""

import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim):
        """
        Args:
            dim (int): Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Create sinusoidal embeddings for timesteps.

        Args:
            time (torch.Tensor): Tensor of shape (batch_size,) with timestep values

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class NoisePredictor(nn.Module):
    """Linear model for predicting noise epsilon_theta(x_t, t)."""

    def __init__(
        self,
        data_dim=2,
        hidden_dim=128,
        time_embed_dim=32,
        num_layers=3
    ):
        """
        Initialize noise prediction network.

        Args:
            data_dim (int): Dimension of input data (2 for 2D points)
            hidden_dim (int): Hidden layer dimension
            time_embed_dim (int): Timestep embedding dimension
            num_layers (int): Number of hidden layers
        """
        super().__init__()

        self.data_dim = data_dim
        self.time_embed_dim = time_embed_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        # Main network
        layers = []
        input_dim = data_dim + time_embed_dim

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (predict noise with same dimension as data)
        layers.append(nn.Linear(hidden_dim, data_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t):
        """
        Predict noise epsilon from noisy data x_t at timestep t.

        Args:
            x_t (torch.Tensor): Noisy data of shape (batch_size, data_dim)
            t (torch.Tensor): Timesteps of shape (batch_size,) with values in [0, num_timesteps-1]

        Returns:
            torch.Tensor: Predicted noise of shape (batch_size, data_dim)
        """
        # Encode timestep
        t_emb = self.time_mlp(t)

        # Concatenate data and timestep embedding
        x_combined = torch.cat([x_t, t_emb], dim=-1)

        # Predict noise
        epsilon_pred = self.net(x_combined)

        return epsilon_pred


if __name__ == "__main__":
    # Test the model
    print("Testing NoisePredictor model...")

    # Create model
    model = NoisePredictor(
        data_dim=2,
        hidden_dim=128,
        time_embed_dim=32,
        num_layers=3
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Test forward pass
    batch_size = 16
    x_t = torch.randn(batch_size, 2)
    t = torch.randint(0, 1000, (batch_size,))

    epsilon_pred = model(x_t, t)

    print(f"\nForward pass test:")
    print(f"  Input x_t shape: {x_t.shape}")
    print(f"  Input t shape: {t.shape}")
    print(f"  Output epsilon shape: {epsilon_pred.shape}")
    print(f"  ✓ Model working correctly!")
