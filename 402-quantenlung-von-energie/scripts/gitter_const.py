#!python3
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from scipy.optimize import curve_fit
import propeller as p
import std

std.bullshit.this_is_fucking_stupid_no_one_actually_gives_a_fuck()

eb_defaults = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


def main():
    data = np.transpose(np.loadtxt(argv[1]))
    color = data[0]
    color_err = 0.005
    d2r = 2 * np.pi / 360
    alpha = p.ev(140, 0.5) * d2r  # was const
    beta = (p.ev(data[1] + 140 - 180, 0.5)) * d2r

    order = 1  # we think

    angle_term = np.sin(alpha) + np.sin(beta)
    angle_term = angle_term / order

    angle_term, angle_err = p.ve(angle_term)

    res, cov = curve_fit(lambda x, a: a * x, angle_term, color)
    err = np.sqrt(np.diag(cov))
    print(f"g = {res[0]} +- {err[0]} nm")
    # print(f"b = {res[1]}")

    xrange = np.linspace(
                         min(angle_term) - 0.075,
                         max(angle_term) + 0.075, 1000)

    goodness = std.goodness_of_fit(color, angle_term * res[0])
    print(goodness)

    gpp = p.ev(color, color_err) / p.ev(angle_term, angle_err)
    gratig_const_per_point, gpp_err = p.ve(gpp)
    print(f"g = {gratig_const_per_point}")
    print(f"g_err = {gpp_err}")
    g_avg = sum(gpp / len(gpp))
    print(f"g_avg = {~g_avg} +- {p.error(g_avg)}")

    label = f"$R^2 = {round(goodness, 2)}$, $g = ({round(res[0])}\\pm{round(err[0])})$ nm"
    plt.plot(xrange, res[0] * xrange, label=label)

    defaults = eb_defaults
    defaults["fmt"] = "x"
    defaults["markersize"] = 5

    plt.errorbar(
                 angle_term, color,
                 yerr=color_err,
                 xerr=angle_err, **defaults)

    plt.legend()
    std.default.plt_pretty(r"$\sin(\alpha) + \sin(\beta)$", r"$\lambda$ / nm")
    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()


if __name__ == "__main__":
    main()
