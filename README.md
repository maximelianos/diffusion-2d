# Forward Diffusion Visualization for 2D Point Space

Visualization of forward diffusion (noise addition) process on 2D point clouds using the sklearn moon dataset with interactive rerun visualization.

* 2D moon dataset in dataset.py
* trainable backward diffusion model
* pytorch dataloader in loader.py
* diffusion.py for forward process

Linear model for noise prediction epsilon_{theta}(x_t, t) in model.py.

Training loop in train.py. Save model checkpoints, plot loss using scripts in loss directory.

Hydra for configuration - conf/config.py contains necessary parameters. Configuration is loaded only in the main function and arguments are passed as dictionary to other classes and functions.

After training, the backward process is visualized in rerun - script inference.py. It loads the checkpoint from disk.










## Project Structure

```
diffusion-example/
├── dataset.py       # Moon dataset generation and plotting utilities
├── diffusion.py     # Forward diffusion model implementation
├── rerun_viz.py     # Rerun visualization functions
├── main.py          # Main script demonstrating forward diffusion
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## What is Forward Diffusion?

Forward diffusion is the process of progressively adding noise to clean data. Given clean data x₀, the process generates increasingly noisy versions x₁, x₂, ..., xₜ according to:

```
x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
```

where:
- α̅_t controls the noise schedule
- ε is Gaussian noise
- As t increases, the data becomes more noisy

This project visualizes how 2D points from the moon dataset gradually become pure noise.

## Components

### 1. Dataset (`dataset.py`)
- Generates 2D moon-shaped point clouds using sklearn
- Provides data normalization utilities
- Saves static matplotlib plots

### 2. Diffusion Model (`diffusion.py`)
- Implements forward diffusion process
- Uses linear beta schedule
- Generates noise trajectories for visualization
- Dataset-independent implementation

### 3. Visualization (`rerun_viz.py`)
- Interactive visualization using rerun library
- Shows point movement over time
- Color gradient: green (clean) → red (noisy)
- Optional individual point tracking

### 4. Main Script (`main.py`)
- Orchestrates the complete pipeline
- Generates moon dataset
- Applies forward diffusion
- Visualizes with rerun

## Usage

### Basic Usage

```bash
python main.py
```

This will:
1. Generate 500 moon dataset samples
2. Initialize diffusion model with 100 timesteps
3. Generate forward diffusion trajectory
4. Launch rerun viewer with interactive visualization

### Command Line Arguments

```bash
python main.py --help
```

Options:
- `--num-samples`: Number of samples to generate (default: 500)
- `--num-timesteps`: Total diffusion timesteps (default: 100)
- `--beta-start`: Starting beta value (default: 0.0001)
- `--beta-end`: Ending beta value (default: 0.02)
- `--viz-steps`: Number of timesteps to visualize (default: 50)
- `--no-rerun`: Disable rerun visualization
- `--save-plots`: Save matplotlib plots to outputs/ directory

### Examples

**More samples and timesteps:**
```bash
python main.py --num-samples 1000 --num-timesteps 200
```

**Save plots without rerun:**
```bash
python main.py --no-rerun --save-plots
```

**Adjust noise schedule:**
```bash
python main.py --beta-start 0.0001 --beta-end 0.05
```

**Visualize more timesteps:**
```bash
python main.py --viz-steps 100
```

## Visualization Guide

The rerun viewer shows:

- **Timeline**: Scrub through timesteps to see noise addition
- **Color coding**:
  - Green: Clean original data (t=0)
  - Yellow/Orange: Intermediate noise levels
  - Red: Maximum noise (t=T)
- **Points**: All samples shown as circles
- **Tracked samples**: A few samples highlighted in cyan for tracking

### Controls

- Use the timeline slider to navigate through timesteps
- Click and drag to rotate the view
- Scroll to zoom
- Space bar to play/pause animation

## How It Works

### Forward Diffusion Formula

```python
# At timestep t, noisy data is computed as:
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

where:
  alpha_bar_t = cumulative product of (1 - beta_t)
  epsilon = standard Gaussian noise
```

### Noise Schedule

The beta schedule controls how quickly noise is added:
- **Linear schedule**: β linearly increases from β_start to β_end
- Early timesteps: Small noise addition, data structure preserved
- Late timesteps: Large noise addition, data becomes Gaussian

### Visualization Timeline

- **t=0**: Original clean moon dataset
- **t=T/4**: Slight blur, structure still visible
- **t=T/2**: Significant noise, structure fading
- **t=3T/4**: Mostly noise, faint structure
- **t=T**: Pure Gaussian noise, no structure

## Testing Individual Components

Each module can be tested independently:

```bash
# Test dataset generation
python dataset.py

# Test diffusion model
python diffusion.py

# Test rerun visualization
python rerun_viz.py
```

## Understanding the Code

### Dataset Generation
```python
from dataset import create_moon_dataset
data = create_moon_dataset(n_samples=500)  # Shape: (500, 2)
```

### Forward Diffusion
```python
from diffusion import DiffusionModel
model = DiffusionModel(num_timesteps=100)

# Single timestep
noisy, noise = model.forward_diffusion(data, t=50)

# Full trajectory
trajectory = model.add_noise_trajectory(data)  # Shape: (100, 500, 2)
```

### Visualization
```python
from rerun_viz import init_rerun, log_forward_diffusion

init_rerun("my_visualization")
log_forward_diffusion(data, trajectory, track_samples=[0, 10, 20])
```

## Mathematical Background

### Variance Schedule

The cumulative noise variance α̅_t follows:
```
α_t = 1 - β_t
α̅_t = ∏(i=0 to t) α_i
```

As t → T:
- α̅_t → 0
- sqrt(α̅_t) → 0 (original data contribution vanishes)
- sqrt(1 - α̅_t) → 1 (noise dominates)

Result: x_T ≈ ε (pure Gaussian noise)

## Requirements

- Python 3.8+
- numpy (numerical operations)
- matplotlib (static plots)
- scikit-learn (moon dataset)
- rerun-sdk (interactive visualization)

## Future Enhancements

Possible extensions:
- [ ] Additional datasets (Swiss roll, circles, custom shapes)
- [ ] Different noise schedules (cosine, sigmoid)
- [ ] 3D point cloud support
- [ ] Animation export (video/GIF)
- [ ] Statistical analysis of noise addition
- [ ] Comparison of different beta schedules

## Notes

- This implementation focuses on forward diffusion only (noise addition)
- For denoising (reverse diffusion), a trained neural network would be needed
- The model is dataset-independent and works with any 2D point cloud
- Rerun provides smooth timeline scrubbing for detailed inspection
