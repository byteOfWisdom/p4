import numpy as np
from sys import argv
import scipy
from matplotlib import pyplot as plt
import std


data = np.transpose(np.loadtxt(argv[1], delimiter=",", skiprows=1))
angle = data[0]
amplitude = data[1]


def cos_sq(x, a, b, c, d):
    return a * np.cos(b * x + c) + d


params, _ = scipy.optimize.curve_fit(cos_sq, angle, amplitude, p0=[0.1, 6 / 360, 0, 0])
# params = std.fit_func(cos_sq, angle, amplitude, p0=[0.1, 6 / 360, 0, 0])


x = np.linspace(0, 360, 10000)
y = cos_sq(x, *params)#.beta)

plt.scatter(angle, amplitude)
plt.plot(x, y)
plt.show()
