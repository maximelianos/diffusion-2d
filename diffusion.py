"""Forward diffusion model for 2D point space."""

import numpy as np


class DiffusionModel:
    """Forward diffusion model - adds noise progressively to data."""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Initialize diffusion model with linear beta schedule.

        Args:
            num_timesteps (int): Number of diffusion timesteps
            beta_start (float): Starting noise variance
            beta_end (float): Ending noise variance
        """
        self.num_timesteps = num_timesteps

        # Create beta schedule (linear)
        self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # Precompute values for efficient forward diffusion
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)

    def forward_diffusion(self, x0, t, noise=None):
        """
        Forward diffusion process: add noise to data.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x0 (np.ndarray): Original data of shape (batch_size, dim)
            t (int): Timestep (0 to num_timesteps-1)
            noise (np.ndarray, optional): Optional pre-generated noise

        Returns:
            tuple: (noisy_data, noise_used) both np.ndarray
        """
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(np.float32)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        noisy_data = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return noisy_data, noise

    def add_noise_trajectory(self, x0, timesteps=None):
        """
        Generate trajectory of forward diffusion process.

        Args:
            x0 (np.ndarray): Original data of shape (batch_size, dim)
            timesteps (np.ndarray, optional): Array of timesteps to sample, or None for all timesteps

        Returns:
            np.ndarray: Array of shape (num_timesteps, batch_size, dim) with noisy versions
        """
        if timesteps is None:
            timesteps = np.arange(self.num_timesteps)

        trajectory = np.zeros((len(timesteps), *x0.shape), dtype=np.float32)

        for i, t in enumerate(timesteps):
            noisy_data, _ = self.forward_diffusion(x0, t)
            trajectory[i] = noisy_data

        return trajectory


if __name__ == "__main__":
    # Test the diffusion model
    print("Testing DiffusionModel...")

    # Initialize model
    model = DiffusionModel(num_timesteps=50)
    print(f"Initialized model with {model.num_timesteps} timesteps")

    # Test forward diffusion
    x0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    xt, noise = model.forward_diffusion(x0, t=25)
    print(f"\nForward diffusion at t=25:")
    print(f"  x0 shape: {x0.shape}")
    print(f"  xt shape: {xt.shape}")
    print(f"  noise shape: {noise.shape}")

    # Test trajectory generation
    trajectory = model.add_noise_trajectory(x0)
    print(f"\nNoise trajectory:")
    print(f"  Shape: {trajectory.shape}")
    print(f"  (timesteps, samples, dimensions)")
    print(f"  Range at t=0: [{trajectory[0].min():.3f}, {trajectory[0].max():.3f}]")
    print(f"  Range at t=49: [{trajectory[-1].min():.3f}, {trajectory[-1].max():.3f}]")
