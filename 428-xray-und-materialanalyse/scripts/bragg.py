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
    #

    return energies_ev, counts


# def fitted_spectrum(x, n, *params):
#     return sum(
#         [
#             gaussian(x, params[i], params[i + 1], params[i + 2])
#             for i in range(0, 3 * n, 3)
#         ]
#     ) + background(x, params[3 * n], params[3 * n + 1])


def fit_peaks(file: str, number, pure=False, verbose=True):
    init_guess = []  # for gaussian params

    xrange = np.linspace(7, 19, 5000)
    #    linear_fit = [8, 1000]
    linear_fit = [160, -900]
    gauss1 = [100, 7.4, 0.06]
    gauss2 = [600, 8.4, 0.1]
    gauss3 = [1600, 9.7, 0.3]
    gauss4 = [800, 11.3, 0.15]
    gauss0 = [80, 11.8, 0.15]
    gauss5 = [500, 9.9, 0.2]

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

    init_guess = gauss0 + gauss1 + gauss2 + gauss3 + gauss4 + gauss5 + linear_fit

    if verbose == True:
        print("init. guesses:", init_guess)

    # get relevant data
    energies_ev, counts = convert(file)
    _, e_ev_err = p.ve(energies_ev)
    _, c_err = p.ve(counts)
    std.default.plt_pretty(f"Energie [eV]", "Intensität [1/s]")
    plt.errorbar(
        ~energies_ev,
        ~counts,
        xerr=e_ev_err,
        yerr=c_err,
        label="Messdaten der unbekannten Anode",
        **std.default.error_bar_def,
    )
    plt.legend(loc="best")
    plt.savefig("../figs/bragg_alldata.pdf")
    plt.show()

    energies_range = energies_ev[(7e3 <= energies_ev) & (12e3 >= energies_ev)]
    counts_range = counts[(7e3 <= energies_ev) & (12e3 >= energies_ev)]

    energies_range_kev = energies_range * 1e-3

    _, e_err = p.ve(energies_range_kev)
    _, c_err = p.ve(counts_range)

    std.default.plt_pretty("Energie [keV]", "Intensität [1/s]")
    plt.errorbar(
        ~energies_range_kev,
        ~counts_range,
        xerr=e_err,
        yerr=c_err,
        **std.default.error_bar_def,
        label="Messdaten",
    )

    # fit data to spectrum func
    fit, cov = scipy.optimize.curve_fit(
        spectrum_func(number),
        ~energies_range_kev,
        ~counts_range,
        p0=init_guess,
        maxfev=9999,
    )

    err = np.sqrt(np.diag(cov))

    if verbose == True:
        # fit parameters
        for i in range(0, number * 3, 3):
            print("A:", fit[i], "+-", err[i])
            print("mu:", fit[i + 1], "+-", err[i + 1])
            print("sigma:", fit[i + 2], "+-", err[i + 2])
        print("a:", fit[-2], "+-", err[-2])
        print("b:", fit[-1], "+-", err[-1])

    # goodness of fit via reduced chi square:
    goodness = round(
        std.goodness_of_fit(
            ~counts_range, spectrum_func(number)(~energies_range_kev, *fit)
        ),
        3,
    )

    fitted = spectrum_func(number)(~energies_range_kev, ~counts_range, *fit)
    chi_square = round(std.reduced_chi_2(~counts_range, fitted, fit), 3)
    if verbose == True:
        print(rf"$\Chi^2_r$:", chi_square)

    fit_range = np.linspace(7, 12, 5000)

    # plot individual fitted gaussians + background

    plt.plot(
        fit_range,
        spectrum_func(number)(fit_range, *fit),
        label=rf"Anpassungsfkt., $\chi^2_r$={str(chi_square).replace('.', ',')}, $R^2$={str(goodness).replace('.', ',')}",
        linestyle="solid",
    )
    if pure is True:
        plt.legend(loc="best")

        plt.savefig("../figs/bragg_fitting_data.pdf")
        return fit, err

    plt.plot(
        fit_range,
        gaussian(fit_range, fit[0], fit[1], fit[2]),
        linestyle="--",
        linewidth=0.8,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[3], fit[4], fit[5]),
        linestyle="--",
        linewidth=0.8,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[6], fit[7], fit[8]),
        linestyle="--",
        linewidth=0.8,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[9], fit[10], fit[11]),
        linestyle="--",
        linewidth=0.8,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[12], fit[13], fit[14]),
        linestyle="--",
        linewidth=0.8,
    )
    plt.plot(
        fit_range,
        gaussian(fit_range, fit[15], fit[16], fit[17]),
        linestyle="--",
        linewidth=0.8,
    )
    plt.plot(
        fit_range,
        background(fit_range, fit[-2], fit[-1]),
        linestyle="solid",
        linewidth=0.8,
        label="Untergrund",
    )

    plt.legend(loc="best", fontsize="x-small")
    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()

    return fit, err


def back_to_wave(file):
    fit, err = fit_peaks(file, 6, verbose=False)
    e1 = p.ev(fit[1], err[1])
    e2 = p.ev(fit[4], err[4])
    e3 = p.ev(fit[7], err[7])
    e4 = p.ev(fit[10], err[10])
    e5 = p.ev(fit[13], err[13])
    e6 = p.ev(fit[16], err[16])

    energies = np.array([e1, e2, e3, e4, e5, e6])
    energies_ev = energies * 1e3
    wavelengths = []

    ref_energies = np.array(
        [
            p.ev(11609.4, 4.4),
            p.ev(7399.1, 1.7),
            8397.6,
            9672.35,
            11285.9,
            9961.5,
        ]
    )

    for i in range(len(energies_ev)):
        value, err = p.ve(energies_ev[i])
        print("Peakenergie:", energies_ev[i].format(), "eV")
        print(
            "Referenzenergie:",
            [ref_energies[i].format() if i < 2 else ref_energies[i]],
        )

    wavelength = lambda energy: (scipy.constants.h * scipy.constants.c) / (
        energy * scipy.constants.e
    )

    for i in energies_ev:
        wavelengths.append(wavelength(i))

    wavelengths = np.array(wavelengths)
    wavelengths_nm = wavelengths * 1e9

    diff_energies = energies_ev - ref_energies
    for i in diff_energies:
        val, err = p.ve(i)
        sigma = True
        if err - val <= 1:
            sigma = True
        elif val - err > 1:
            sigma = False

        print("Differenz Energie:", i.format(), "eV", sigma)

    ref_wavelengths = wavelength(ref_energies)
    ref_wavelengths_nm = ref_wavelengths * 1e9

    diff_wavelengths = -ref_wavelengths_nm + wavelengths_nm

    for i in range(len(ref_wavelengths_nm)):
        if i == 0 or i == 1:
            print("Referenzwellenlänge:", ref_wavelengths_nm[i].format(), "nm")
            print("Wellenlänge:", wavelengths_nm[i].format(), "nm")
            print("Diff:", diff_wavelengths[i].format(), "nm")
        else:
            print("Referenzwellenlänge:", ref_wavelengths_nm[i], "nm")
            print("Wellenlänge:", wavelengths_nm[i].format(), "nm")
            print("Diff:", diff_wavelengths[i].format(), "nm")


def main():
    # convert(argv[1])
    # fit_peaks(argv[1], 6, pure=True, verbose=True)
    back_to_wave(argv[1])
    return


if __name__ == "__main__":
    main()
