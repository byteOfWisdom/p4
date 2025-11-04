#!python3
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from scipy.optimize import curve_fit
import propeller as p


def piecwise_linear(x, d, a, alpha, b):
    mask_a = 1 * (x < d)
    mask_b = 1 * (x >= d)
    beta = (a - alpha) * d + b
    return mask_a * (x * a + b) + mask_b * (x * alpha + beta)


eb_defaults = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


def main():
    file = argv[1]
    data = np.transpose(np.loadtxt(file, delimiter=",", skiprows=1))
    voltage = data[0]  # Volt
    current = (data[1] - data[1][0]) * 1e-10  # Ampere
    y = np.sqrt(current)

    param, cov = curve_fit(piecwise_linear, voltage, y, p0=[-1, 0, 0.1, 0])

    errs = np.sqrt(np.diag(cov))
    delim = p.ev(param[0], errs[0])
    a = p.ev(param[1], errs[1])
    alpha = p.ev(param[2], errs[2])
    b = p.ev(param[3], errs[3])
    beta = (a - alpha) * delim + b
    voltage_zero = - beta / alpha
    print(f"{float(voltage_zero)} +- {p.error(voltage_zero)}")

    xrange = np.linspace(min(voltage) - 0.1, max(voltage) + 0.1, 10000)
    plt.plot(xrange, piecwise_linear(xrange, *param))
    plt.errorbar(voltage, y, np.sqrt(10e-13), xerr=0.1, **eb_defaults)
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel("Gegenspannung U / V")
    plt.ylabel(r"Photostrom $\sqrt{I - I_0}$ / $\sqrt{A}$")
    plt.show()


if __name__ == "__main__":
    main()
