#!python3
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from scipy.optimize import curve_fit
import scipy as sp
import propeller as p
import std

std.bullshit.this_is_fucking_stupid_no_one_actually_gives_a_fuck()

eb_defaults = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


def main():
    data = np.transpose(np.loadtxt(argv[1]))
    color = data[0]
    alpha = np.deg2rad(140)  # was const
    beta = np.deg2rad(data[1] + 140 - 180)

    order = 1  # we think

    angle_term = np.sin(alpha) + np.sin(beta)
    angle_term = angle_term / order

    res, cov = curve_fit(lambda x, a, b: a * x + b, color, angle_term)
    err = np.sqrt(np.diag(cov))
    print(f"g = {1 / res[0]} +- {1 / err[0]} nm")

    xrange = np.linspace(min(color) - 20, max(color) + 20, 1000)

    # chi_sq, p_value = sp.stats.chisquare(res[0] * color, f_exp=angle_term)
    # print(f"chi2 = {chi_sq}, p = {p_value}")

    gratig_const_per_point = color / angle_term
    print(f"g = {gratig_const_per_point}")

    plt.plot(xrange, res[0] * xrange)

    plt.errorbar(
                 color, angle_term,
                 xerr=color * 0.01,
                 yerr=angle_term * 0.01, **eb_defaults)
    # plt.grid(which="major")
    # plt.grid(which="minor", linestyle=":", linewidth=0.5)
    # plt.gca().minorticks_on()
    # plt.xlabel(r"$\lambda$ / nm")
    # plt.ylabel(r"$\sin(\alpha) + \sin(\beta)$")
    std.default.plt_pretty(r"$\lambda$ / nm", r"$\sin(\alpha) + \sin(\beta)$")
    plt.show()


if __name__ == "__main__":
    main()
