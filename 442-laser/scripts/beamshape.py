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

    print(~dist_rel)

    # theoretical gaussian values
    waist = np.sqrt(wavelength / np.pi * np.sqrt(length * (curve - length)))
    rayleigh = np.pi * (waist**2) / wavelength
    print("w0:", waist.format())
    print("z_0:", rayleigh.format())
    # theoretical gaussian
    xrange = np.linspace(0.4, 0.55, 500)

    gaussian_theo = lambda x: waist * np.sqrt(1 + (x / rayleigh) ** 2)

    plt.plot(xrange, gaussian_theo(xrange), label=r"$\omega(z)_\text{theo}$")
    plt.plot(
        xrange, gaussian_theo(xrange) * 1.18, label=r"$\omega(z)_\text{theo,korr}$"
    )

    # fitting gaussian to data
    #
    rayleigh_meas = lambda a: (np.pi * (a**2)) / wavelength
    gaussian_meas = lambda x, w, y: (w * np.sqrt(1 + ((x - y) / rayleigh_meas(w)) ** 2))

    fit, cov = curve_fit(gaussian_meas, ~dist_rel, ~radii, p0=[0.003, 0], maxfev=20000)
    print("w_0,meas:", fit[0])
    print("b:", fit[1])

    plt.plot(xrange, gaussian_meas(xrange, fit[0], fit[1]), label="fit")
    # plt.plot(xrange, gaussian_theo(xrange) + fit[1], label="theo mit offset")
    plt.scatter(~dist_rel, ~radii, marker="x")

    # linear = lambda x, a, b: a * x + b
    # xvalues = gaussian_x(dist_rel) #incorrect!! this gives the theoretical xvalues

    # fit_parm, cov = curve_fit(linear, xvalues, radii, p0=[1, -1]) #incorrect fitting func
    # err = np.sqrt(np.diag(cov))
    # print("omega_0:", fit_parm[0], "+-", err[0], "b:", fit_parm[1], "+-", err[1])

    # f = linear(xrange, fit_parm[0], fit_parm[1])
    # theo = linear(xrange, waist, 0)

    # goodness = round(
    #     std.goodness_of_fit(~radii, linear(~xvalues, fit_parm[0], fit_parm[1])), 3
    # )
    # print("R^2:", goodness)

    # _, xerr = p.ve(xvalues)
    # _, yerr = p.ve(radii)

    # plt.plot(xrange, f, color="darkblue")
    # plt.plot(xrange, theo, color="red")

    # plt.errorbar(
    #     ~xvalues,
    #     ~radii,
    #     xerr=xerr,
    #     yerr=yerr,
    #     **std.default.error_bar_def,
    #     label=rf"$R^2$={goodness}",
    #     color="forestgreen",
    # )

    plt.legend(loc="best")

    std.default.plt_pretty("z / m", r"$\omega(z)$ / m")

    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()


if __name__ == "__main__":
    main()
