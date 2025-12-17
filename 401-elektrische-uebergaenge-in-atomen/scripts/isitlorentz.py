import numpy as np
import std
from matplotlib import pyplot as plt
import scipy

def double_l(x, a0, a1, x0, x1, gamma0, gamma1, c):
    return std.lorentz_curve(x, a0, x0, gamma0) + std.lorentz_curve(x, a1, x1, gamma1) + c

data = np.transpose(np.loadtxt("../data/B_ort_calib.csv", delimiter=",", skiprows=1))
x, y = data[0], data[1]

p0 = [0.45, 0.45, 0.197, 0.203, 0.01, 0.01, 0.05]
# params, _ = std.fit_func(double_l, x, y, p0=p0)
params, _ = scipy.optimize.curve_fit(double_l, x, y, p0=p0)


xrange = np.linspace(min(x), max(x), 10000)
plt.scatter(x, y, marker="x")
plt.plot(xrange, double_l(xrange, *params))
plt.show()
