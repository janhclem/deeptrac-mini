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

logfile = "training.log"
training_stats = np.genfromtxt(logfile, delimiter=",", skip_header=1, dtype=float).T

num_iter = len(training_stats[1])

plt.figure()
print(training_stats)
plt.scatter(range(num_iter), training_stats[1])
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss Over Iterations")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.show()
