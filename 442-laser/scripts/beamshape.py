from sys import argv

import numpy as np
import propeller as p
import scipy as sp
import std
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

std.bullshit.ger()


def get_waists(filename):
    offset = 14.8e-2
    data = np.transpose(np.loadtxt(filename, skiprows=1, delimiter=","))
    dists = p.ev(data[0], 5e-3) - p.ev(offset, 5e-3)
    widths = p.ev(data[1], 0.01e-3)
    return dists, widths


def main():
    # params of setup
    if len(argv) > 3:
        wavelength = float(argv[3])
    else:
        wavelength = 632.8e-9  # literature for HeNe laser transition
    length = p.ev(51.3e-2, 5e-3)
    curve = 1

    dists, widths = get_waists(argv[1])
    radii = widths / 2
    dist_rel = length - dists  # from plane mirror

    for i in dist_rel:
        print(rf"\num{ {i.format()} }")

    # print(~dist_rel)
    # print(~radii)

    xrange = np.linspace(0.44, 0.52, 500)

    # theoretical gaussian
    waist = np.sqrt(wavelength / np.pi * np.sqrt(length * (curve - length)))
    rayleigh = np.pi * (waist**2) / wavelength
    print("w0,theo:", waist.format())
    print("z_0,theo:", rayleigh.format())

    gaussian_theo = lambda x: waist * np.sqrt(1 + (x / rayleigh) ** 2)

    # plt.plot(
    #     xrange, gaussian_theo(xrange) * 1.18, label=r"$\omega(z)_\text{theo,korr}$"
    # )

    # fitting pure gaussian to data
    rayleigh_meas = lambda a: (np.pi * (a**2)) / wavelength
    gaussian_meas = lambda x, w, y: (w * np.sqrt(1 + ((x - y) / rayleigh_meas(w)) ** 2))

    fit_1, cov_1 = curve_fit(
        gaussian_meas, ~dist_rel, ~radii, p0=[0.003, 0], maxfev=20000
    )
    err_1 = np.sqrt(np.diag(cov_1))
    print("w_0,meas:", fit_1[0], "+-", err_1[0])
    print("y (z offset):", fit_1[1], "+-", err_1[1])

    # corrected gaussian model w/ intensity parameter
    gaussian_theo_corr = lambda x, y: y * (~waist * np.sqrt(1 + (x / ~rayleigh) ** 2))

    r, r_err = p.ve(radii)
    d, d_err = p.ve(dist_rel)

    fit_2, cov_2 = curve_fit(gaussian_theo_corr, d, r)
    err_2 = np.sqrt(np.diag(cov_2))
    print("a:", fit_2[0], "+-", err_2[0])

    # goodness tests
    goodness_1 = round(std.goodness_of_fit(r, gaussian_meas(d, fit_1[0], fit_1[1])), 3)
    print("R^2,meas:", goodness_1)
    goodness_2 = round(std.goodness_of_fit(r, gaussian_theo_corr(d, fit_2[0])), 3)
    print("R^2,corr theo:", goodness_2)

    # pure data fit
    plt.plot(
        xrange,
        gaussian_meas(xrange, fit_1[0], fit_1[1]),
        label=rf"$R^2$={goodness_1}",
        color="darkblue",
    )
    # parameter corrected gaussian fit
    plt.plot(
        xrange,
        gaussian_theo_corr(xrange, fit_2[0]),
        label=r"$a\cdot\omega(z)_\text{theo}$: " + rf"$R^2=${goodness_2}",
        linestyle="--",
        color="orange",
    )
    # data w errors
    plt.errorbar(
        ~dist_rel,
        ~radii,
        xerr=d_err,
        yerr=r_err,
        **std.default.error_bar_def,
        color="forestgreen",
    )
    # theoretical gaussian
    plt.plot(
        xrange,
        gaussian_theo(xrange),
        label=r"$\omega(z)_\text{theo}$",
        color="darkviolet",
    )

    plt.legend(loc="best")

    std.default.plt_pretty("z / m", r"$\omega(z)$ / m")

    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()


if __name__ == "__main__":
    main()
