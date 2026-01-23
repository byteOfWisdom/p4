from mimetypes import init
from sys import argv

import numpy as np
import propeller as p
import scipy
import std
from matplotlib import pyplot as plt

h = scipy.constants.h
c = scipy.constants.c


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(((x - mu) / sigma) ** 2))


def background(x, a, b):
    #    return -a * ((x - 15) ** 2) + b
    return a * x + b


def spectrum_func(n):
    multigaussian = lambda x, *params: sum(
        [
            gaussian(x, params[i], params[i + 1], params[i + 2])
            for i in range(0, 3 * n, 3)
        ]
    ) + background(x, params[3 * n], params[3 * n + 1])
    return np.vectorize(multigaussian)


def get_data(file: str):
    data = np.transpose(
        np.loadtxt(
            file,
            skiprows=1,
            delimiter="\t",
            converters=lambda c: float(c.replace(",", ".")),
        )
    )
    return data[0], data[1]


def degtorad(deg_angle):
    rad_angle = deg_angle * (np.pi / 180)
    return rad_angle


def convert(file: str, order: int = 1):
    angles, counts = get_data(file)
    angles = p.ev(angles, 0.05)  # fehler aus winkelschritt/2
    counts = p.ev(counts, counts * 0.03)
    order = order
    abstand = 564.00e-12 / 2  # netzebenenabstand

    wavelength = lambda theta: 2 * abstand * np.sin(degtorad(theta)) / order
    energies = h * c / wavelength(angles)
    energies_ev = energies / scipy.constants.e

    # std.default.plt_pretty(r"Wellenlänge $\lambda$ [m]", "Intensität [1/s]")
    # _ = plt.plot(
    #     wavelength(angles),
    #     counts,
    #     label="Messung unbekannte Anode",
    #     color="hotpink",
    #     linewidth=1.2,
    # )
    # plt.show()

    return energies_ev, counts


# def fitted_spectrum(x, n, *params):
#     return sum(
#         [
#             gaussian(x, params[i], params[i + 1], params[i + 2])
#             for i in range(0, 3 * n, 3)
#         ]
#     ) + background(x, params[3 * n], params[3 * n + 1])


def fit_peaks(file: str, number):
    init_guess = []  # for gaussian params

    xrange = np.linspace(7, 19, 5000)
    #    linear_fit = [8, 1000]
    linear_fit = [160, -900]
    gauss1 = [100, 7.4, 0.06]
    gauss2 = [600, 8.4, 0.1]
    gauss3 = [1600, 9.7, 0.3]
    gauss4 = [800, 11.3, 0.15]
    gauss0 = [80, 11.8, 0.15]

    if number == 5:
        manual = (
            gaussian(xrange, *gauss0)
            + gaussian(xrange, *gauss1)
            + gaussian(xrange, *gauss2)
            + gaussian(xrange, *gauss3)
            + gaussian(xrange, *gauss4)
            + background(xrange, *linear_fit)
        )
    # other peak number options go here
    # _ = plt.plot(xrange, manual)

    init_guess = gauss0 + gauss1 + gauss2 + gauss3 + gauss4 + linear_fit

    print("init. guesses:", init_guess)

    # get relevant data
    energies_ev, counts = convert(file)

    energies_range = energies_ev[(7e3 <= energies_ev) & (12e3 >= energies_ev)]
    counts_range = counts[(7e3 <= energies_ev) & (12e3 >= energies_ev)]

    energies_range_kev = energies_range * 1e-3

    _, e_err = p.ve(energies_range_kev)
    _, c_err = p.ve(counts_range)

    plt.plot(~energies_range_kev, ~counts_range, label="Messdaten")

    # fit data to spectrum func
    fit, cov = scipy.optimize.curve_fit(
        spectrum_func(number),
        ~energies_range_kev,
        ~counts_range,
        p0=init_guess,
        maxfev=9999,
    )

    err = np.sqrt(np.diag(cov))

    # fit parameters
    for i in range(0, number * 3, 3):
        print("A:", fit[i], "+-", err[i])
        print("mu:", fit[i + 1], "+-", err[i + 1])
        print("sigma:", fit[i + 2], "+-", err[i + 2])
    print("a:", fit[-2], "+-", err[-2])
    print("b:", fit[-1], "+-", err[-1])

    # goodness of fit:
    goodness = round(
        std.goodness_of_fit(~counts_range, spectrum_func(number)(~energies_range_kev, *fit)),
        3,
    )
    print("R^2:", goodness)

    # print(fit)

    fit_range = np.linspace(7, 12, 5000)
    # fitted_func = fitted_spectrum(fit_range, 5, *init_guess)

    # plot individual fitted gaussians + background
    plt.plot(
        fit_range,
        spectrum_func(number)(fit_range, *fit),
        label="Fitgerade",
        linewidth=0.9,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[0], fit[1], fit[2]),
        linestyle="--",
        linewidth=0.6,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[3], fit[4], fit[5]),
        linestyle="--",
        linewidth=0.6,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[6], fit[7], fit[8]),
        linestyle="--",
        linewidth=0.6,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[9], fit[10], fit[11]),
        linestyle="--",
        linewidth=0.6,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[12], fit[13], fit[14]),
        linestyle="--",
        linewidth=0.6,
    )
    plt.plot(
        fit_range,
        background(fit_range, fit[-2], fit[-1]),
        linestyle="--",
        linewidth=0.6,
    )

    plt.legend()
    plt.show()
    return


def main():
    fit_peaks(argv[1], 5)
    return


if __name__ == "__main__":
    main()
