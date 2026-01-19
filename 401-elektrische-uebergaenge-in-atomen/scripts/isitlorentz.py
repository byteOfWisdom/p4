import numpy as np
import std
from matplotlib import pyplot as plt
import scipy


def inv_sq(x, a, x0, gamma):
    return a / (1 + gamma * (x - x0)**2)


def double_l(x, a0, a1, x0, x1, gamma0, gamma1, c):
    # return std.lorentz_curve(x, a0, x0, gamma0) + std.lorentz_curve(x, a1, x1, gamma1) + c
    return inv_sq(x, a0, x0, gamma0) + inv_sq(x, a1, x1, gamma1) + c


data = np.transpose(np.loadtxt("../data/B_ort_calib.csv", delimiter=",", skiprows=1))
x, y = data[0], data[1]

p0 = [0.45, 0.45, 0.196, 0.203, 100000, 100000, 0.05]
# params, _ = std.fit_func(double_l, x, y, p0=p0)
params, _ = scipy.optimize.curve_fit(double_l, x, y, p0=p0, maxfev=99999)

print(params)

xrange = np.linspace(min(x), max(x), 10000)
plt.scatter(x, y, marker="x")
# plt.plot(xrange, double_l(xrange, *p0))
plt.plot(xrange, double_l(xrange, *params))
std.default.plt_pretty("y ort / m", "B feld an hall sonde / T")
plt.show()
