"""
Solution to differential equation dy / dx = y:

dy / y = dx
int(1 / y * dy) = int(dx)
ln y + C1 = x + C2
e^(ln y + C) = e^x
y = e^x * C
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_curves(t, data):
    """
    Arguments:
        t: array of length n
        data: [arrays of length n]
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # plot train curve
    for y in data:
        ax.plot(t, y, 'b--', label='Training Loss')

    #ylim = max(train_loss.mean(), val_loss.mean())
    #ax.set_ylim((0, ylim))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Differential equation solution')
    #ax.legend()
    ax.grid(True)

    # Save the plot
    output_path = "solution.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Solution plot saved to {output_path}")

def main():
    # store trajectories
    data = []
    
    # solve the equation y' = y
    n_lines = 5
    for y0 in np.linspace(0, 2, n_lines):
        # initial conditions
        x = 0
        y = y0
        traj = [y]
        
        # x step size
        dx = 0.0001
        steps = int(3 / dx)
        for step in range(steps - 1):
            # make step
            y1 = y * (dx + 1)
            y = y1
            traj.append(y)
        data.append(traj)
        
    x = np.linspace(0, 1, steps)
    plot_curves(x, data)

if __name__ == "__main__":
    main()