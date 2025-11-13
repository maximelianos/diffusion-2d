"""Training script for backward diffusion model - independent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Import custom modules
from model import NoisePredictor
from loader import create_dataloader


class DiffusionTrainer:
    """Trainer for diffusion model."""

    def __init__(self, config: dict):
        """
        Initialize trainer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Set device
        self.device = torch.device(
            config['training']['device'] if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Set random seed
        torch.manual_seed(config['training']['seed'])
        np.random.seed(config['training']['seed'])

        # Create diffusion schedule
        self._setup_diffusion_schedule()

        # Create model
        self.model = NoisePredictor(
            data_dim=config['model']['data_dim'],
            hidden_dim=config['model']['hidden_dim'],
            time_embed_dim=config['model']['time_embed_dim'],
            num_layers=config['model']['num_layers']
        ).to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {num_params:,} parameters")

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Create dataloader
        self.dataloader, self.norm_params = create_dataloader(
            n_samples=config['dataset']['n_samples'],
            batch_size=config['dataloader']['batch_size'],
            noise=config['dataset']['noise'],
            random_state=config['dataset']['random_state'],
            normalize=config['dataset']['normalize'],
            shuffle=config['dataloader']['shuffle'],
            num_workers=config['dataloader']['num_workers']
        )

        # Loss criterion
        self.criterion = nn.MSELoss()

        # Training history
        self.losses = []

        # Create checkpoint directory
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create plot directory
        self.plot_dir = Path(config['training']['plot_dir'])
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def _setup_diffusion_schedule(self):
        """Setup diffusion schedule (beta, alpha, alpha_bar)."""
        num_timesteps = self.config['diffusion']['num_timesteps']
        beta_start = self.config['diffusion']['beta_start']
        beta_end = self.config['diffusion']['beta_end']

        # Create beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute values for efficient forward diffusion
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion: q(x_t | x_0).

        Args:
            x0: Clean data of shape (batch_size, data_dim)
            t: Timesteps of shape (batch_size,)

        Returns:
            Tuple of (noisy_data, noise)
        """
        # Sample noise
        noise = torch.randn_like(x0)

        # Get sqrt_alpha_bar and sqrt_one_minus_alpha_bar for timesteps t
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)

        # Apply forward diffusion
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def train_step(self, batch: torch.Tensor) -> float:
        """
        Perform single training step.

        Args:
            batch: Batch of clean data

        Returns:
            Loss value
        """
        batch = batch.to(self.device)
        batch_size = batch.shape[0]

        # Sample random timesteps
        t = torch.randint(
            0,
            self.config['diffusion']['num_timesteps'],
            (batch_size,),
            device=self.device
        )

        # Forward diffusion (add noise)
        x_t, noise = self.forward_diffusion(batch, t)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # Calculate loss
        loss = self.criterion(noise_pred, noise)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, step: int, filename: str = None):
        """
        Save model checkpoint.

        Args:
            step: Current training step
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"model_step_{step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'config': self.config,
            'norm_params': self.norm_params
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Saved checkpoint to {checkpoint_path}")

    def plot_losses(self, step: int):
        """
        Plot and save loss curves.

        Args:
            step: Current training step
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot step losses
        ax.plot(self.losses, alpha=0.6, linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

        # Add moving average
        if len(self.losses) > 100:
            window = 100
            moving_avg = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.losses)), moving_avg, 'r-', linewidth=2, label='Moving Avg (100)')
            ax.legend()

        plt.tight_layout()

        # Save plot
        plot_path = self.plot_dir / f"loss_step_{step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved loss plot to {plot_path}")

    def train(self):
        """Run full training loop."""
        num_steps = self.config['training']['num_steps']
        save_every = self.config['training']['save_every']
        plot_every = self.config['training']['plot_every']

        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Number of steps: {num_steps}")
        print(f"Batch size: {self.config['dataloader']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print("=" * 60 + "\n")

        self.model.train()
        step = 0

        # Create infinite dataloader
        from itertools import cycle
        data_iter = cycle(self.dataloader)

        pbar = tqdm(range(num_steps), desc="Training")
        for step in pbar:
            # Get next batch
            batch = next(data_iter)

            # Train step
            loss = self.train_step(batch)
            self.losses.append(loss)

            # Update progress bar
            if step % 10 == 0:
                avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else np.mean(self.losses)
                pbar.set_postfix({'loss': f'{loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})

            # Save checkpoint
            if (step + 1) % save_every == 0:
                print(f"\nStep {step+1}/{num_steps}")
                self.save_checkpoint(step + 1)

            # Plot losses
            if (step + 1) % plot_every == 0:
                self.plot_losses(step + 1)

        # Save final checkpoint
        self.save_checkpoint(num_steps, filename="model_final.pt")
        self.plot_losses(num_steps)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total steps: {num_steps}")
        print(f"Final loss: {self.losses[-1]:.6f}")
        print(f"Average loss (last 100): {np.mean(self.losses[-100:]):.6f}")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        print(f"Loss plots saved in: {self.plot_dir}")
        print("=" * 60)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Convert OmegaConf to dict for easier handling
    config = OmegaConf.to_container(cfg, resolve=True)

    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Create trainer
    trainer = DiffusionTrainer(config)

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
