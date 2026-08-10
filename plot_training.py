"""
Plot training loss from the training log file.

This script reads the training log CSV file and plots the loss over
training iterations on a logarithmic scale.

Copyright (C) 2026 Jan Clemens

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Usage
-----
    python plot_training.py

This will read training.log (default) and display a plot of loss vs iteration.
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import generic_filter

def moving_average(x, window):
    """Simple centered moving average; shorter window at the edges."""
    if window <= 1 or len(x) == 0:
        return x
    kernel = np.ones(window) / window
    # 'same' keeps output length equal to input length
    return np.convolve(x, kernel, mode="same")

def moving_median(x, window):
    """Centered moving median with shorter window at edges."""
    if window <= 1 or len(x) == 0:
        return x
    
    # Pad with edge values to handle boundaries
    pad_size = window // 2
    x_padded = np.pad(x, pad_size, mode='edge')
    
    # Apply rolling median
    result = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = i
        end = start + window
        result[i] = np.median(x_padded[start:end])
    
    return result

logfile = "./log/training.log"
training_stats = np.genfromtxt(logfile, delimiter=",", skip_header=1, dtype=float).T

num_iter = len(training_stats[1])

# Plot 1: RMSE with both moving average and moving median
plt.figure(figsize=(5,4))
plt.scatter(range(num_iter), training_stats[-1], c="r", s=0.1, alpha=0.5)
plt.plot(range(num_iter-200), moving_average(training_stats[-1], 100)[100:-100], c="m", lw=3, label="Mean(100)")
plt.plot(range(num_iter-200), moving_median(training_stats[-1], 100)[100:-100], c="g", lw=2, label="Median(100)")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("RMSE")
plt.legend(loc='best')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig("./training_loss.png", dpi=300)

# Plot 2: Mass balance with both moving average and moving median
plt.figure(figsize=(5,4))
plt.scatter(range(num_iter), np.abs(training_stats[-2]), c="b", s=0.1, alpha=0.5)
plt.plot(range(num_iter), moving_average(np.abs(training_stats[-2]), 1000), c="c", lw=3, label="MA(1000)")
plt.plot(range(num_iter), moving_median(np.abs(training_stats[-2]), 1000), c="orange", lw=2, label="Median(1000)")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Abs. Mass balance per particle")
plt.title("Mass balance")
plt.legend(loc='best')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig("./mass_balance.png", dpi=300)

# Plot 3: R with both moving average and moving median
plt.figure(figsize=(5,4))
plt.scatter(range(num_iter), 1-training_stats[-1]**2, c="r", s=0.1, alpha=0.5)
plt.plot(range(num_iter-200), moving_average(1-training_stats[-1]**2, 100)[100:-100], c="m", lw=3, label="Mean(100)")
plt.plot(range(num_iter-200), moving_median(1-training_stats[-1]**2, 100)[100:-100], c="g", lw=2, label="Median(100)")
#plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.ylim(0,1)
plt.title("Pearson Correlation Coefficient Estimation")
plt.legend(loc='best')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig("./training_r.png", dpi=300)










