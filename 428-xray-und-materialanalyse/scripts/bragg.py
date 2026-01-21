from sys import argv

import numpy as np
import scipy
import std
from matplotlib import pyplot as plt

h = scipy.constants.h
c = scipy.constants.c


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


def convert(file: str, order: int = 1):
    angles, counts = get_data(file)
    std.default.plt_pretty(r"Winkel $\beta$", "Intensität")
    order = order
    abstand = 564.00e-12 / 2
    wavelength = lambda theta: 2 * abstand * np.sin(np.deg2rad(theta)) / order
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
    #
    std.default.plt_pretty(f"Energie [keV]", "Intensität")
    plt.plot(
        energies_ev * 1e-3,
        counts,
        label="Messung unbekannte Anode",
        color="hotpink",
        linewidth=1.2,
    )
    plt.show()
    return energies_ev, counts


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(((x - mu) / sigma) ** 2))


def background(x, a, b):
    return -a * ((x - 15) ** 2) + b


def spectrum_func(n):
    multigaussian = lambda x, *params: sum(
        [
            gaussian(x, params[i], params[i + 1], params[i + 2])
            for i in range(0, 3 * n, 3)
        ]
    ) + background(x, params[3 * n], params[3 * n + 1])
    return multigaussian


def fitted_spectrum(x, n, *params):
    return sum(
        [
            gaussian(x, params[i], params[i + 1], params[i + 2])
            for i in range(0, 3 * n, 3)
        ]
    ) + background(x, params[3 * n], params[3 * n + 1])


def fit_peaks(file: str):
    init_guess = []  # for gaussian params

    xrange = np.linspace(0, 50, 500)
    linear_fit = [8, 1000]
    gauss0 = [100, 5.5, 0.06]
    gauss1 = [100, 7, 0.06]
    gauss2 = [600, 8, 0.1]
    gauss3 = [1000, 9.5, 0.1]
    gauss4 = [800, 12, 0.15]

    manual = (
        gaussian(xrange, *gauss0)
        + gaussian(xrange, *gauss1)
        + gaussian(xrange, *gauss2)
        + gaussian(xrange, *gauss3)
        + gaussian(xrange, *gauss4)
        + background(xrange, *linear_fit)
    )
    plt.plot(xrange, manual)
    energies_ev, counts = convert(file)

    init_guess = gauss0 + gauss1 + gauss2 + gauss3 + gauss4 + linear_fit
    print(init_guess)

    energies_ev, counts = convert(file)
    # TO DO: data slicing
    energies_range = []
    counts_range = []
    for i in range(len(energies_ev)):
        if energies_ev[i] <= 35e3:
            energies_range.append(energies_ev[i])
            counts_range.append(counts[i])
    plt.plot(energies_range, counts_range)
    fit, cov = scipy.optimize.curve_fit(
        spectrum_func(5),
        energies_ev,
        counts,
        p0=init_guess,
        maxfev=9999,
    )
    # fit parameter error here

    err = np.sqrt(np.diag(cov))
    print(fit)

    fit_range = np.linspace(0, 35e3, len(energies_range))
    fitted_func = fitted_spectrum(fit_range, 5, init_guess)

    plt.plot(fit_range, fitted_spectrum(fit_range, 5, *init_guess), label="fitted")
    plt.legend()
    plt.show()
    return


def main():
    fit_peaks(argv[1])
    return


if __name__ == "__main__":
    main()
