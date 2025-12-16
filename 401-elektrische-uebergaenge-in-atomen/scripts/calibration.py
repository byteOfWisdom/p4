from sys import argv

import numpy as np
import propeller as p
import scipy
import std
from matplotlib import pyplot as plt

std.bullshit.ger()


def get_curve(file):
    data = np.transpose(np.loadtxt(file, delimiter=";", skiprows=2))
    # convert to SI units
    current = p.ev(data[0], 0.2)  # placeholder error values
    field = p.ev(data[1], 8) * 1e-3
    return current, field


def main():
    std.default.plt_pretty("Spulenstrom / A", "Magnetfeld B / T")
    cubic = lambda x, a, b, c, d: (a * (x**3)) + (b * (x**2)) + (c * x) + d
    for i in range(len(argv)):
        if i >= 1:
            currents, fields = get_curve(argv[i])
            params, cov = scipy.optimize.curve_fit(cubic, ~currents, ~fields)
            errs = np.sqrt(np.diag(cov))
            print(
                "a:",
                params[0],
                "+-",
                errs[0],
                "b:",
                params[1],
                "+-",
                errs[1],
                "c:",
                params[2],
                "+-",
                errs[2],
                "d:",
                params[3],
                "+-",
                errs[3],
            )

            eb_param = std.default.error_bar_def
            _, curr_err = p.ve(currents)
            _, field_err = p.ve(fields)
            goodness = round(std.goodness_of_fit(~fields, cubic(~currents, *params)), 4)
            print("R^2,meas:", goodness)
            # errors are placeholders
            plt.errorbar(
                ~currents,
                ~fields,
                xerr=curr_err,
                yerr=field_err,
                **eb_param,
                label=f"Messung {i}",
            )
            xrange = np.linspace(min(~currents), max(~currents), 1000)
            plt.plot(
                xrange,
                [cubic(x, *params) for x in xrange],
                label=rf"Fit {i}, $R^2$={goodness}",
            )
        else:
            continue
        if i == 1:
            params_1 = params
        if i == 2:
            params_2 = params

    # just to evaluate differences, dont include in final graph
    diff_xrange = np.linspace(-10, 10, 1000)
    cubic_1 = [cubic(x, *params_1) for x in diff_xrange]
    cubic_2 = [cubic(x, *params_2) for x in diff_xrange]
    diff_cubic = np.linspace(0, 1, len(diff_xrange))
    for x in range(len(diff_xrange)):
        diff_cubic[x] = cubic_1[x] - cubic_2[x]
    plt.plot(diff_xrange, diff_cubic, label="Differenz der Fitfunktionen")

    plt.legend(loc="best")
    plt.show()


# def fit_cubic(file):
#     currents, fields = get_curve(file)
#     cubic = lambda x, a, b, c, d: (a * (x**3)) + (b * (x**2)) + (c * x) + d
#     params, cov = scipy.optimize.curve_fit(cubic, currents, fields)
#     return params, cov


if __name__ == "__main__":
    main()
