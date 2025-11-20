from sys import argv

import numpy as np
import scipy as sp
import std
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# import propeller as p

messcolor = "forestgreen"
eb_defaults = {"fmt": " ", "elinewidth": 0.75, "capsize": 2, "color": messcolor}


def main():
    data = np.transpose(np.loadtxt(argv[1], skiprows=1, delimiter=","))
    degrees = data[0]
    voltages = data[1]

    xrange = np.linspace(0, 360, 37)

    max_trans = max(voltages)
    min_trans = min(voltages)
    print(min(voltages))

    dop = (max_trans - min_trans) / (max_trans + min_trans)
    print("degree of polarisation: ", dop)

    init_guess = [
        0,  # a
        6,  # b
        0,  # c
        6 / 360,  # d
    ]

    malus = lambda x, a, b, c, d: a * ((np.cos((d * x) + b)) ** 2) + c

    fit_func, cov = curve_fit(
        malus,
        degrees,
        voltages,
        init_guess,
    )

    f = (fit_func[0] * np.cos(fit_func[3] * (fit_func[1] - degrees)) ** 2) + fit_func[2]

    err = np.sqrt(np.diag(cov))

    goodness = round(std.goodness_of_fit(voltages, f), 3)

    print(
        "a: ",
        fit_func[0],
        "I_0:",
        fit_func[0] + fit_func[2],
        "phi:",
        fit_func[1],
        "c:",
        fit_func[2],
        "d:",
        fit_func[3],
    )

    plt.scatter(degrees, voltages, marker=".", color=messcolor)
    plt.errorbar(degrees, voltages, xerr=5, yerr=voltages * 0.05, **eb_defaults)
    # yerror: 5% of value

    plt.plot(
        xrange,
        f,
        linestyle="-",
        label=f"$R^2 = {goodness}$",
        color="darkblue",
    )

    plt.ylim(-0.01, max(voltages) + 0.02)
    plt.grid(which="major")
    plt.grid(
        which="minor",
        linestyle=":",
        linewidth=0.5,
    )
    plt.gca().minorticks_on()
    plt.legend(loc="upper right")
    plt.xlabel(r"Verdrehungswinkel /$^\circ$")
    plt.ylabel(rf"U / V")

    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()


if __name__ == "__main__":
    main()
