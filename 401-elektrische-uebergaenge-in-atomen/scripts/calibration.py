from sys import argv

import numpy as np
import propeller as p
import scipy
import std
from matplotlib import pyplot as plt

def get_curve(file):
    data = np.transpose(np.loadtxt(file, delimiter=";", skiprows=2))
    # convert to SI units
    current = p.ev(data[0], 0.2)  # placeholder error values
    field = p.ev(data[1], 8) * 1e-3
    return current, field


def calibration_curve(to_print=False,to_plot=True):
    files = ["data/zeeman/a203/Kalibration_01.txt", "data/zeeman/a203/Kalibration_02.txt"]
    std.bullshit.ger()
    std.default.plt_pretty("Spulenstrom I / A", "Magnetfeld B / T")
    cubic = lambda x, a, b, c, d: (a * (x**3)) + (b * (x**2)) + (c * x) + d
    plot_color = iter(["crimson","forestgreen"])
    func_params = []
    func_errs = []
    for i in range(0,2):
        messung = i + 1
        this_color = next(plot_color)
        currents, fields = get_curve("/".join(argv[0].split("/")[:-1]) + "../" + files[i])
        params, cov = scipy.optimize.curve_fit(cubic, ~currents, ~fields)
        errs = np.sqrt(np.diag(cov))
        func_params.append(params)
        func_errs.append(errs)
        if to_print is True:
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
        if to_print is True:
            print("R^2,meas:", goodness)
            # errors are placeholders
        if to_plot is True:
            xrange = np.linspace(min(~currents), max(~currents), 1000)
            plt.plot(
                xrange,
                [cubic(x, *params) for x in xrange],
                label=rf"$R^2_{i+1}$={goodness}",
                color = this_color,
                linestyle = "--"
            )
            plt.errorbar(
                ~currents,
                ~fields,
                xerr=curr_err,
                yerr=field_err,
                **eb_param,
                label=f"Messung {messung}",
                color = this_color,
            )
            plt.legend(loc="best")
        # if i == 1:
        #     params_1 = params
        # if i == 2:
        #     params_2 = params
    return func_params, func_errs

    # just to evaluate differences, dont include in final graph
    # diff_xrange = np.linspace(-10, 10, 1000)
    # cubic_1 = [cubic(x, *params_1) for x in diff_xrange]
    # cubic_2 = [cubic(x, *params_2) for x in diff_xrange]
    # diff_cubic = np.linspace(0, 1, len(diff_xrange))
    # for x in range(len(diff_xrange)):
    #     diff_cubic[x] = cubic_1[x] - cubic_2[x]
    #plt.plot(diff_xrange, diff_cubic, label="Differenz der Fitfunktionen")

def average_field(verbal=False):
    params, errs = calibration_curve(to_plot=False, to_print=False)
    params_1 = p.ev(params[0], errs[0])
    params_2 = p.ev(params[1], errs[1])
    p_1, err_1 = p.ve(params_1)
    p_2, err_2 = p.ve(params_2)
    av_params = (params_1+params_2) / 2
    av_vals, av_errs = p.ve(av_params)
    if verbal is True:
        print("Parameterset 1:", p_1)
        print("Fehler 1:", err_1)
        print("Parameterset 2:", p_2)
        print("Fehler 2:", err_2)
        print("Average Parameter:",av_vals)
        print("Average Fehler:", av_errs)
    cubic = lambda x: (av_params[0] * (x**3)) + (av_params[1] * (x**2)) + (av_params[2] * x) + av_params[3]
    #temp,_ = p.ve(np.vectorize(cubic)(np.linspace(-10, 10, 100)))
    #plt.plot(np.linspace(-10, 10, 100), temp, color="black")
    return av_vals, av_errs

def field_function(current):
    cubic_params, cubic_errs = average_field()
    cubic = (cubic_params[0] * (current**3)) + (cubic_params[1] * (current**2)) + (cubic_params[2] * current) + cubic_params[3]
    return cubic


def main():
    #calibration_curve()
    #average_field(verbal=True)
    print(field_function(5.23))
    if len(argv) >= 2:
        plt.savefig(argv[1])
    else:
        plt.show()


if __name__ == "__main__":
    main()
