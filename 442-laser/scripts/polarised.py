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

    # u_nolaser = 0.0e-3  # correction term  not necessary due to being zero lol

    corrected_voltages = voltages - u_nolaser

    xrange = np.linspace(0, 360, 37)

    # for degree of polarisation

    max_trans = max(voltages)
    min_trans = min(voltages)

    dop = (max_trans - min_trans) / (max_trans + min_trans)
    print("degree of polarisation: ", dop)

    # fitting cos^2 func to data

    # necessary to give start geusses or fit WILL be bad
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

    f = (fit_func[0] * np.cos((fit_func[3] * degrees) + fit_func[1]) ** 2) + fit_func[2]
    err = np.sqrt(np.diag(cov))

    print(
        f"a*cos(d*x+b)^2+c\n",
        "a:",
        fit_func[0],
        "I_0:",
        fit_func[0] + fit_func[2],
        "b:",
        fit_func[1],
        "c:",
        fit_func[2],
        "d:",
        fit_func[3],
    )

    # useful to use coeff. of determination R^2 for linear model fit
    goodness = round(std.goodness_of_fit(voltages, f), 3)

    # plotting data & fit
    plt.scatter(degrees, voltages, marker=".", color=messcolor)
    plt.errorbar(
        degrees,
        voltages,
        xerr=5,
        yerr=voltages * 0.05,
        **eb_defaults,
    )
    # data yerror: 5% of value, xerror: 5 degrees fixed
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
    plt.xlabel(r"Verdrehungswinkel / $^\circ$")
    plt.ylabel(rf"U / V")

    # allow saving to specified file as optional 3rd argv
    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()


if __name__ == "__main__":
    main()
