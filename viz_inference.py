"""Inference script for backward diffusion with rerun visualization."""

import torch
import numpy as np
import rerun as rr
from pathlib import Path
import argparse
from tqdm import tqdm

from model import NoisePredictor


class BackwardDiffusion:
    """Backward diffusion sampler using trained model."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize backward diffusion sampler.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load checkpoint (weights_only=False needed for numpy arrays in checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.norm_params = checkpoint.get('norm_params')

        # Handle both step-based and epoch-based checkpoints
        if 'step' in checkpoint:
            print(f"Loaded checkpoint from step {checkpoint['step']}")
            if 'losses' in checkpoint and len(checkpoint['losses']) > 0:
                avg_loss = np.mean(checkpoint['losses'][-100:])
                print(f"Training loss (avg last 100): {avg_loss:.6f}")
        else:
            # Legacy epoch-based checkpoint
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0) + 1}")
            if 'epoch_losses' in checkpoint and len(checkpoint['epoch_losses']) > 0:
                print(f"Training loss: {checkpoint['epoch_losses'][-1]:.6f}")

        # Setup diffusion schedule
        self._setup_diffusion_schedule()

        # Create and load model
        self.model = NoisePredictor(
            data_dim=self.config['model']['data_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            time_embed_dim=self.config['model']['time_embed_dim'],
            num_layers=self.config['model']['num_layers']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print("✓ Model loaded successfully")

    def _setup_diffusion_schedule(self):
        """Setup diffusion schedule (beta, alpha, alpha_bar)."""
        num_timesteps = self.config['diffusion']['num_timesteps']
        beta_start = self.config['diffusion']['beta_start']
        beta_end = self.config['diffusion']['beta_end']

        # Create beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute values for efficient backward diffusion
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - self.alphas)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @torch.no_grad()
    def backward_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single backward diffusion step: p(x_{t-1} | x_t).

        Args:
            x_t: Noisy data at timestep t, shape (batch_size, data_dim)
            t: Current timestep

        Returns:
            x_{t-1}: Less noisy data at timestep t-1
        """
        batch_size = x_t.shape[0]

        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        epsilon_pred = self.model(x_t, t_tensor)

        # Get diffusion parameters
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        sqrt_alpha_t = self.sqrt_alphas[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]

        # Compute mean of p(x_{t-1} | x_t)
        # mean = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_pred)
        mean = (x_t - ((1 - alpha_t) / sqrt_one_minus_alpha_bar_t) * epsilon_pred) / sqrt_alpha_t

        if t > 0:
            # Add noise for t > 0
            noise = torch.randn_like(x_t)
            beta_t = self.betas[t]
            sigma_t = torch.sqrt(beta_t)
            x_t_minus_1 = mean + sigma_t * noise
        else:
            # No noise at t=0
            x_t_minus_1 = mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, n_samples: int, num_steps: int = None) -> np.ndarray:
        """
        Generate samples using backward diffusion.

        Args:
            n_samples: Number of samples to generate
            num_steps: Number of backward steps (default: num_timesteps)

        Returns:
            Trajectory of shape (num_steps+1, n_samples, data_dim)
        """
        if num_steps is None:
            num_steps = self.config['diffusion']['num_timesteps']

        data_dim = self.config['model']['data_dim']

        # Start from pure noise at t=T
        x_t = torch.randn(n_samples, data_dim, device=self.device)

        # Store trajectory
        trajectory = [x_t.cpu().numpy()]

        # Backward diffusion
        timesteps = np.linspace(num_steps - 1, 0, num_steps, dtype=int)

        print(f"\nGenerating {n_samples} samples with {num_steps} backward steps...")
        for t in tqdm(timesteps, desc="Backward diffusion"):
            x_t = self.backward_step(x_t, t)
            trajectory.append(x_t.cpu().numpy())

        trajectory = np.array(trajectory)

        # Denormalize if needed
        if self.norm_params is not None:
            print("Denormalizing samples...")
            trajectory = self._denormalize(trajectory)

        return trajectory

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data back to original range.

        Args:
            data: Normalized data

        Returns:
            Denormalized data
        """
        min_vals = self.norm_params['min']
        max_vals = self.norm_params['max']
        return (data + 1) / 2 * (max_vals - min_vals) + min_vals


def init_rerun(app_name: str = "backward_diffusion_2d") -> None:
    """
    Initialize rerun recording.

    Args:
        app_name: Name of the rerun application
    """
    rr.init(app_name, spawn=True)


def log_backward_diffusion(
    trajectory: np.ndarray,
    entity_base: str = "backward_diffusion"
) -> None:
    """
    Log backward diffusion process (denoising) to rerun.

    Args:
        trajectory: Array of shape (num_timesteps+1, num_samples, 2)
        entity_base: Base entity path
    """
    num_timesteps, num_samples, _ = trajectory.shape

    print(f"\nLogging {num_timesteps} timesteps to rerun...")

    # Log each timestep
    for t in tqdm(range(num_timesteps), desc="Logging to rerun"):
        rr.set_time_sequence("timestep", t)

        # Color gradient: red (noisy) → green (clean)
        progress = t / max(num_timesteps - 1, 1)
        red = int(255 * (1 - progress))
        green = int(255 * progress)
        colors = np.array([[red, green, 0, 255]] * num_samples, dtype=np.uint8)

        rr.log(
            f"{entity_base}/points",
            rr.Points2D(
                trajectory[t],
                colors=colors,
                radii=0.02
            )
        )


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Backward diffusion inference with rerun visualization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_final.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of backward diffusion steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--no-rerun", action="store_true", help="Disable rerun visualization")
    args = parser.parse_args()

    print("=" * 60)
    print("Backward Diffusion Inference")
    print("=" * 60)

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print(f"  Please train the model first using: python train.py")
        return

    # Load model and create sampler
    print(f"\nLoading checkpoint: {checkpoint_path}")
    sampler = BackwardDiffusion(checkpoint_path=str(checkpoint_path), device=args.device)

    # Generate samples
    trajectory = sampler.sample(n_samples=args.n_samples, num_steps=args.num_steps)

    print(f"\n✓ Generated trajectory shape: {trajectory.shape}")
    print(f"  (timesteps, samples, dimensions)")

    # Visualize with rerun
    if not args.no_rerun:
        print("\nInitializing rerun visualization...")
        init_rerun("backward_diffusion_2d")

        log_backward_diffusion(trajectory, entity_base="backward_diffusion")

        print("\n" + "=" * 60)
        print("✓ Visualization complete!")
        print("  Check the rerun viewer to see the backward diffusion process")
        print("  Points move from noise (red) → clean data (green)")
        print("=" * 60)
    else:
        print("\nRerun visualization disabled")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Initial noise (t={args.num_steps}):")
    print(f"  Mean: {trajectory[0].mean(axis=0)}")
    print(f"  Std:  {trajectory[0].std(axis=0)}")
    print(f"\nGenerated data (t=0):")
    print(f"  Mean: {trajectory[-1].mean(axis=0)}")
    print(f"  Std:  {trajectory[-1].std(axis=0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
