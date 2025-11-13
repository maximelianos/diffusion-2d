"""PyTorch DataLoader for diffusion training."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset import create_moon_dataset, normalize_data


class DiffusionDataset(Dataset):
    """Dataset for diffusion model training."""

    def __init__(self, data: np.ndarray):
        """
        Initialize dataset with 2D point data.

        Args:
            data: Array of shape (n_samples, 2) containing 2D points
        """
        self.data = torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single data point.

        Args:
            idx: Index of the data point

        Returns:
            Data point of shape (2,)
        """
        return self.data[idx]


def create_dataloader(
    n_samples: int = 1000,
    batch_size: int = 128,
    noise: float = 0.05,
    random_state: int = 42,
    normalize: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
) -> tuple[DataLoader, dict | None]:
    """
    Create PyTorch DataLoader for moon dataset.

    Args:
        n_samples: Number of samples to generate
        batch_size: Batch size for training
        noise: Standard deviation of Gaussian noise for dataset
        random_state: Random seed for reproducibility
        normalize: Whether to normalize data to [-1, 1]
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (dataloader, normalization_params or None)
    """
    # Create dataset
    data = create_moon_dataset(n_samples=n_samples, noise=noise, random_state=random_state)

    # Normalize if requested
    norm_params = None
    if normalize:
        data, norm_params = normalize_data(data)

    # Create dataset and dataloader
    dataset = DiffusionDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader, norm_params


if __name__ == "__main__":
    # Test the dataloader
    print("Testing DiffusionDataset and DataLoader...")

    # Create dataloader
    dataloader, norm_params = create_dataloader(
        n_samples=1000,
        batch_size=32,
        normalize=True
    )

    print(f"\nDataLoader created:")
    print(f"  Total samples: {len(dataloader.dataset)}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Number of batches: {len(dataloader)}")

    # Test iteration
    print(f"\nTesting batch iteration:")
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"  Batch shape: {batch.shape}")
            print(f"  Batch dtype: {batch.dtype}")
            print(f"  Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
            print(f"  ✓ DataLoader working correctly!")
        break

    # Check normalization params
    if norm_params:
        print(f"\nNormalization parameters:")
        print(f"  Min: {norm_params['min']}")
        print(f"  Max: {norm_params['max']}")
