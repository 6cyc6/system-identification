import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0.001
variance = 0.0000001
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma)*(sigma * np.sqrt(2*np.pi)))
plt.show()