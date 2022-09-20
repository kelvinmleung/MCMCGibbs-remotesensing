import numpy as np
import matplotlib.pyplot as plt


a = np.random.normal(size=1000)

lower = min(a)
upper = max(a)
bins = np.linspace(lower, upper, 10)
plt.hist(a, bins, alpha=0.5)