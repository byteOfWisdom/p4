from sys import argv

import numpy as np
import propeller as p
import scipy
import std
from matplotlib import pyplot as plt

from bragg import convert, gaussian, get_data


def double_gaussian():
    return np.vectorize(
        lambda x, *params: gaussian(x, *params[0:3])
        + gaussian(x, *params[3:-1])
        + params[-1]
    )  # just to be safe explicitly offset


def handle_data(file):
    angles, counts = get_data(file)
    angles = p.ev(angles, 0.005)
    counts = p.ev(counts, counts * 0.05)  # larger error est. due to tiny incidence
    _, a_err = p.ve(angles)
    _, c_err = p.ve(counts)
    std.default.plt_pretty("Winkel", "Intensität")
    plt.errorbar(
        ~angles,
        ~counts,
        xerr=a_err,
        yerr=c_err,
        **std.default.error_bar_def,
        label="Messdaten",
    )
    plt.show()

    return


# handles conversion to energy, defaults to 4th order for molybden tube
def energy_data(file: str, order: int = 4):
    energies_ev, counts = convert(file, order=order)
    energies_kev = energies_ev * 1e-3
    _, e_err = p.ve(energies_kev)
    _, c_err = p.ve(counts)

    std.default.plt_pretty("Energie [keV]", "Intensität [1/s]")
    _ = plt.errorbar(
        ~energies_kev,
        ~counts,
        xerr=e_err,
        yerr=c_err,
        **std.default.error_bar_def,
        label="Messdaten",
    )
    plt.legend(loc="best")
    plt.savefig("../figs/bragg_mo_alldata.pdf")
    plt.show()
    return energies_kev, counts


# fits count(energy) data to double gaussian
def fit_curve(file):
    energies_kev, counts = energy_data(file)

    counts = p.ev(~counts, ~counts * 0.1)

    energies_range = energies_kev[(17 <= energies_kev) & (energies_kev <= 17.8)]
    counts_range = counts[(17 <= energies_kev) & (energies_kev <= 17.8)]
    _, e_err = p.ve(energies_range)
    _, c_err = p.ve(counts_range)

    # std.default.plt_pretty("Energie [keV]", "Intensität")
    # plt.errorbar(
    #     ~energies_range,
    #     ~counts_range,
    #     xerr=e_err,
    #     yerr=c_err,
    #     **std.default.error_bar_def,
    #     label="Messdatenausschnitt",
    # )
    # plt.plot(~energies_range, ~counts_range)

    xrange = np.linspace(17, 17.8, 10000)

    init_guess = []
    gauss1 = [1.9, 17.28, 0.05]
    gauss2 = [2.4, 17.38, 0.1]
    offset = [0.5]
    init_guess = gauss1 + gauss2 + offset
    # plt.plot(xrange, gaussian(xrange, *gauss1))
    # plt.plot(xrange, gaussian(xrange, *gauss2))
    # plt.legend()
    # plt.show()

    fit, cov = scipy.optimize.curve_fit(
        double_gaussian(), ~energies_range, ~counts_range, p0=init_guess, maxfev=9999
    )

    err = np.sqrt(np.diag(cov))

    print("A1:", fit[0], "+-", err[0])
    print("mu1:", fit[1], "+-", err[1])
    print("sigma1:", fit[2], "+-", err[2])
    print("A2:", fit[3], "+-", err[3])
    print("mu2:", fit[4], "+-", err[4])
    print("sigma2:", fit[5], "+-", err[5])
    print("offset:", fit[-1], "+-", err[-1])

    # goodness of fit
    goodness = round(
        std.goodness_of_fit(~counts_range, double_gaussian()(~energies_range, *fit)),
        3,
    )
    print("R^2:", goodness)

    fitted = double_gaussian()(~energies_range, *fit)
    chi_square = round(std.reduced_chi_2(~counts_range, fitted, fit), 3)
    print(rf"$\Chi^2_r$:", chi_square)

    std.default.plt_pretty("Energie [keV]", "Intensität [1/s]")
    plt.errorbar(
        ~energies_range,
        ~counts_range,
        xerr=e_err,
        yerr=c_err,
        **std.default.error_bar_def,
        label=rf"Messdaten (Ausschnitt)",
    )
    plt.plot(
        xrange,
        double_gaussian()(xrange, *fit),
        label=rf"Anpassungsfkt., $\chi^2_r$={str(chi_square).replace('.', ',')}, $R^2$={str(goodness).replace('.', ',')}",
        linewidth=1,
    )
    # plt.plot(xrange,gaussian(xrange,*fit[0:3]),label="Gauss 1")
    # plt.plot(xrange,gaussian(xrange,*fit[3:]),label="Gauss 2")
    plt.legend(loc="best", fontsize="smaller")
    if len(argv) >= 3:
        plt.savefig(argv[2])
    else:
        plt.show()
    return fit, err


def energy_to_wavelength(file):
    fit, err = fit_curve(file)

    # energy values & spacing
    energy1 = p.ev(fit[1], err[1])  # keV
    energy2 = p.ev(fit[4], err[4])  # keV

    energy1_ev = energy1 * 1e3
    energy2_ev = energy2 * 1e3

    diff_e = energy2_ev - energy1_ev
    _, diff_e_err = p.ve(diff_e)

    # e1 = e2_ref!
    # https://xdb.lbl.gov/Section1/Table_1-2.pdf
    ref_e1 = 17479.34  # eV
    ref_e2 = 17374.30  # eV
    ref_diff_e = ref_e1 - ref_e2

    badness_perc_e = 1 - diff_e / ref_diff_e
    worst_bad_e = 1 - (diff_e + diff_e_err) / ref_diff_e
    least_bad_e = 1 - (diff_e - diff_e_err) / ref_diff_e

    print("E diff (eV):", ~diff_e, "+-", diff_e_err)
    print("Ref E diff: (eV)", ref_diff_e)

    print(
        "Energie-Abweichung:",
        round(~badness_perc_e, 3),
        round(~worst_bad_e, 3),
        round(~least_bad_e, 3),
    )

    # wavelength values + spacing
    wavelength = lambda energy: (scipy.constants.h * scipy.constants.c) / (
        energy * scipy.constants.e
    )

    ref_w1 = wavelength(ref_e1)
    ref_w2 = wavelength(ref_e2)
    ref_w_diff_nm = (ref_w2 - ref_w1) * 1e9

    w1 = wavelength(energy1_ev)
    w2 = wavelength(energy2_ev)
    w1_nm = w1 * 1e9
    w2_nm = w2 * 1e9

    print("ref w1:", ref_w1 * 1e9, "nm")
    print("ref w2:", ref_w2 * 1e9, "nm")

    _, w1_nm_err = p.ve(w1_nm)
    _, w2_nm_err = p.ve(w2_nm)
    diff_w = w1 - w2
    diff_w_nm = diff_w * 1e9
    _, diff_w_err = p.ve(diff_w)
    _, diff_w_nm_err = p.ve(diff_w_nm)

    badness_perc_w = 1 - diff_w_nm / ref_w_diff_nm
    worst_bad_w = 1 - (diff_w_nm + diff_w_nm_err) / ref_w_diff_nm
    least_bad_w = 1 - (diff_w_nm - diff_w_nm_err) / ref_w_diff_nm

    print("w1:", ~w1_nm, "+-", w1_nm_err, "nm")
    print("w2:", ~w2_nm, "+-", w2_nm_err, "nm")
    print("diff w:", ~diff_w_nm, "+-", diff_w_nm_err, "nm")
    print("ref w diff:", ref_w_diff_nm, "nm")
    print(
        "Wellenlänge Abweichung:",
        round(~badness_perc_w, 3),
        round(~worst_bad_w, 3),
        round(~least_bad_w, 3),
    )
    return


def main():
    energy_to_wavelength(argv[1])

    return


if __name__ == "__main__":
    main()
