"""Dataset generation and visualization for 2D point space."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from pathlib import Path


def create_moon_dataset(n_samples: int = 1000, noise: float = 0.05, random_state: int = 42) -> np.ndarray:
    """
    Create 2D moon dataset.

    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, 2) containing 2D points
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float32)


def save_dataset_plot(data: np.ndarray, save_path: str = "dataset.png", title: str = "Moon Dataset") -> None:
    """
    Save a 2D scatter plot of the dataset.

    Args:
        data: Array of shape (n_samples, 2) containing 2D points
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Normalize data to [-1, 1] range.

    Args:
        data: Array of shape (n_samples, 2)

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Normalize to [-1, 1]
    normalized = 2 * (data - min_vals) / (max_vals - min_vals) - 1

    params = {
        'min': min_vals,
        'max': max_vals
    }

    return normalized, params


def denormalize_data(data: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize data back to original range.

    Args:
        data: Normalized array of shape (n_samples, 2)
        params: Normalization parameters from normalize_data

    Returns:
        Denormalized data
    """
    return (data + 1) / 2 * (params['max'] - params['min']) + params['min']


if __name__ == "__main__":
    # Test the dataset generation
    data = create_moon_dataset(n_samples=1000)
    print(f"Generated dataset with shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

    # Save plot
    save_dataset_plot(data, "dataset.png")

    # Test normalization
    normalized, params = normalize_data(data)
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Test denormalization
    denormalized = denormalize_data(normalized, params)
    print(f"Denormalized matches original: {np.allclose(data, denormalized)}")
