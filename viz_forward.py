"""Visualization functionality using rerun library."""

import rerun as rr
import numpy as np
from typing import Optional
import hydra
from omegaconf import DictConfig


def init_rerun(app_name: str = "diffusion_2d") -> None:
    """
    Initialize rerun recording.

    Args:
        app_name: Name of the rerun application
    """
    rr.init(app_name, spawn=True)


def log_points(
    points: np.ndarray,
    entity_path: str = "points",
    colors: Optional[np.ndarray] = None,
    radii: Optional[float] = None
) -> None:
    """
    Log 2D points to rerun.

    Args:
        points: Array of shape (n_points, 2)
        entity_path: Rerun entity path
        colors: Optional colors for points, shape (n_points, 3) or (n_points, 4)
        radii: Optional radius for points
    """
    rr.log(
        entity_path,
        rr.Points2D(
            points,
            colors=colors,
            radii=radii
        )
    )


def log_forward_diffusion(
    x0: np.ndarray,
    noisy_trajectory: np.ndarray,
    entity_base: str = "forward_diffusion",
    track_samples: Optional[list] = None
) -> None:
    """
    Log forward diffusion process (adding noise) with optional individual sample tracking.

    Args:
        x0: Original clean data of shape (num_samples, 2)
        noisy_trajectory: Array of shape (num_timesteps, num_samples, 2)
        entity_base: Base entity path
        track_samples: Optional list of sample indices to track individually
    """
    num_timesteps, num_samples, _ = noisy_trajectory.shape

    # Log original data at t=0
    rr.set_time_sequence("timestep", 0)
    colors_clean = np.array([[0, 255, 0, 255]] * len(x0), dtype=np.uint8)
    log_points(x0, entity_path=f"{entity_base}/points", colors=colors_clean, radii=0.02)

    # Log noisy versions at each timestep
    for t in range(num_timesteps):
        rr.set_time_sequence("timestep", t + 1)

        # Color gradient: green (clean) → red (noisy)
        progress = t / max(num_timesteps - 1, 1)
        red = int(255 * progress)
        green = int(255 * (1 - progress))
        colors = np.array([[red, green, 0, 255]] * num_samples, dtype=np.uint8)

        log_points(
            noisy_trajectory[t],
            entity_path=f"{entity_base}/points",
            colors=colors,
            radii=0.02
        )

    # Track individual samples if requested
    if track_samples is not None:
        for idx in track_samples:
            if idx >= num_samples:
                continue

            # Log original position
            rr.set_time_sequence("timestep", 0)
            rr.log(
                f"{entity_base}/tracked/sample_{idx}",
                rr.Points2D(
                    x0[idx:idx+1],
                    colors=[[0, 255, 255, 255]],
                    radii=0.04
                )
            )

            # Log trajectory
            for t in range(num_timesteps):
                rr.set_time_sequence("timestep", t + 1)
                progress = t / max(num_timesteps - 1, 1)
                red = int(255 * progress)
                green = int(255 * (1 - progress))

                rr.log(
                    f"{entity_base}/tracked/sample_{idx}",
                    rr.Points2D(
                        noisy_trajectory[t, idx:idx+1],
                        colors=[[red, green, 255, 255]],
                        radii=0.04
                    )
                )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main visualization function with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    from dataset import create_moon_dataset
    from diffusion import DiffusionModel

    print("=" * 60)
    print("Forward Diffusion Visualization with Rerun")
    print("=" * 60)

    # Create dataset
    print("\nCreating dataset...")
    data = create_moon_dataset(
        n_samples=cfg.dataset.n_samples,
        noise=cfg.dataset.noise,
        random_state=cfg.dataset.random_state
    )
    print(f"✓ Generated {len(data)} samples")

    # Initialize diffusion model
    print("\nInitializing diffusion model...")
    model = DiffusionModel(
        num_timesteps=cfg.diffusion.num_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end
    )
    print(f"✓ Model initialized with {model.num_timesteps} timesteps")

    # Generate trajectory
    print("\nGenerating forward diffusion trajectory...")
    trajectory = model.add_noise_trajectory(data)
    print(f"✓ Generated trajectory: {trajectory.shape}")

    # Initialize rerun
    print("\nInitializing rerun visualization...")
    init_rerun("forward_diffusion_2d")

    # Log to rerun
    print("Logging forward diffusion to rerun...")
    log_forward_diffusion(data, trajectory, entity_base="diffusion")

    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("Check the rerun viewer to see the forward diffusion process")
    print("=" * 60)


if __name__ == "__main__":
    main()
