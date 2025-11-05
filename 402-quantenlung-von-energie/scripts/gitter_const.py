#!python3
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from scipy.optimize import curve_fit
import propeller as p


eb_defaults = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


def main():
    data = np.transpose(np.loadtxt(argv[1]))
    color = data[0]
    alpha = np.deg2rad(140)  # was const
    beta = np.deg2rad(data[1])

    order = 1  # we think

    angle_term = np.sin(alpha) + np.sin(beta)
    angle_term = angle_term / order

    plt.errorbar(
                 color, angle_term,
                 xerr=color * 0.01,
                 yerr=angle_term * 0.01, **eb_defaults)
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(r"$\lambda$ / nm")
    plt.ylabel(r"$\sin(\alpha) + \sin(\beta)$")
    plt.show()


if __name__ == "__main__":
    main()
