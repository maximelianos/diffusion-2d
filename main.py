"""Main script to demonstrate forward diffusion on 2D moon dataset."""

import numpy as np
import argparse
from dataset import create_moon_dataset, save_dataset_plot
from diffusion import DiffusionModel
from viz_forward import init_rerun, log_forward_diffusion


def main():
    parser = argparse.ArgumentParser(description="Forward diffusion on 2D moon dataset")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--num-timesteps", type=int, default=100, help="Number of diffusion timesteps")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="Starting beta value")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Ending beta value")
    parser.add_argument("--viz-steps", type=int, default=50, help="Number of timesteps to visualize")
    parser.add_argument("--no-rerun", action="store_true", help="Disable rerun visualization")
    parser.add_argument("--save-plots", action="store_true", help="Save matplotlib plots")
    args = parser.parse_args()

    print("=" * 60)
    print("Forward Diffusion on 2D Moon Dataset")
    print("=" * 60)

    # Step 1: Create dataset
    print("\n[1/3] Creating moon dataset...")
    data = create_moon_dataset(n_samples=args.num_samples)
    print(f"  ✓ Generated {len(data)} samples")
    print(f"  ✓ Data range: [{data.min():.3f}, {data.max():.3f}]")

    if args.save_plots:
        save_dataset_plot(data, "outputs/original_dataset.png", "Original Moon Dataset")
        print("  ✓ Saved plot to outputs/original_dataset.png")

    # Step 2: Initialize diffusion model
    print(f"\n[2/3] Initializing diffusion model...")
    model = DiffusionModel(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    print(f"  ✓ Model initialized with {model.num_timesteps} timesteps")
    print(f"  ✓ Beta schedule: {args.beta_start} → {args.beta_end}")

    # Step 3: Generate forward diffusion trajectory
    print(f"\n[3/3] Generating forward diffusion trajectory...")

    # Select timesteps to visualize
    viz_timesteps = np.linspace(0, args.num_timesteps - 1, args.viz_steps, dtype=int)
    trajectory = model.add_noise_trajectory(data, timesteps=viz_timesteps)
    print(f"  ✓ Generated trajectory: {trajectory.shape}")
    print(f"  ✓ Visualizing {args.viz_steps} timesteps")

    # Visualize with rerun
    if not args.no_rerun:
        print(f"\nVisualizing with rerun...")
        init_rerun("forward_diffusion_2d")

        print("  ✓ Logging forward diffusion...")
        log_forward_diffusion(
            data,
            trajectory,
            entity_base="diffusion"
        )

        print("\n" + "=" * 60)
        print("✓ Visualization complete!")
        print("  Check the rerun viewer to see points move from clean → noisy")
        print("=" * 60)
    else:
        print("\nRerun visualization disabled")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"Original data (t=0):")
    print(f"  Mean: {data.mean(axis=0)}")
    print(f"  Std:  {data.std(axis=0)}")
    print(f"\nNoisy data (t={args.num_timesteps-1}):")
    print(f"  Mean: {trajectory[-1].mean(axis=0)}")
    print(f"  Std:  {trajectory[-1].std(axis=0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
