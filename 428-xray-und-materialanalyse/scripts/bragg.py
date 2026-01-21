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
    return a * np.exp(((x - mu) / sigma) ** 2)


def linear_background(x, a, b):
    return a * x + b


def spectrum_func(n: int):
    multigaussian = lambda x, *params: sum(
        [
            gaussian(x, params[i], params[i + 1], params[i + 2])
            for i in range(0, 3 * n, 3)
        ]
    ) + linear_background(x, params[3 * n + 3], params[3 * n + 4])
    return multigaussian


def fit_peaks(file: str):
    init_guess = []  # for gaussian params

    linear_fit = [160, -50]
    gauss_easy = [1, 1, 1]
    init_guess = (gauss_easy * 6) + linear_fit
    print(init_guess)

    energies_ev, counts = convert(file)
    # TO DO: data slicing
    #
    fit, cov = scipy.optimize.curve_fit(
        spectrum_func(6), energies_ev, counts, p0=init_guess
    )

    # fit parameter error here -> issue with fitting func

    err = np.sqrt(np.diag(cov))
    for i in fit:
        print(fit[i])

    return


def main():
    fit_peaks(argv[1])
    return


if __name__ == "__main__":
    main()
