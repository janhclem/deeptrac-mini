import numpy as np
import matplotlib.pylab as plt

logfile="training.log"
training_stats = np.genfromtxt(logfile, delimiter=",", skip_header=1, dtype=float).T

num_iter = len(training_stats[1])

plt.figure()
print(training_stats)
plt.scatter(range(num_iter), training_stats[1])
plt.yscale("log")
plt.show()
