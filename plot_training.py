"""
Plot training loss from the training log file.

This script reads the training log CSV file and plots the loss over
training iterations on a logarithmic scale.

Copyright (c) 2026 Forschungszentrum Juelich GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

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
#plt.yscale("log")
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










