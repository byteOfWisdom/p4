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
    dists, widths = get_waists(argv[1])
    if len(argv) > 3:
        wavelength = float(argv[3])
    else:
        wavelength = 632.8e-9  # literature for HeNe laser transition
    length = p.ev(51.3e-2, 5e-3)
    curve = 1

    dist_rel = length - dists

    waist = np.sqrt(wavelength / np.pi * np.sqrt(length * (curve - length)))
    rayleigh = np.pi * (waist**2) / wavelength
    print("w0:", waist.format())
    print("z_0:", rayleigh.format())

    radii = widths / 2
    # print([x.format() for x in radii])

    # gaussian shape
    linear = lambda x, a, b: a * x + b
    gaussian_x = lambda x: np.sqrt(1 + (x / rayleigh) ** 2)

    xvalues = gaussian_x(dist_rel)

    xrange = np.linspace(0, max(xvalues) + 0.03, 50)

    fit_parm, cov = curve_fit(linear, xvalues, radii, p0=[1, -1])
    err = np.sqrt(np.diag(cov))
    print("omega_0:", fit_parm[0], "+-", err[0], "b:", fit_parm[1], "+-", err[1])

    f = linear(xrange, fit_parm[0], fit_parm[1])
    theo = linear(xrange, waist, 0)

    goodness = round(
        std.goodness_of_fit(~radii, linear(~xvalues, fit_parm[0], fit_parm[1])), 3
    )
    print("R^2:", goodness)

    _, xerr = p.ve(xvalues)
    _, yerr = p.ve(radii)

    plt.plot(xrange, f, color="darkblue")
    plt.plot(xrange, theo, color="red")

    # plt.scatter(~xvalues, ~radii, marker="x")
    plt.errorbar(
        ~xvalues,
        ~radii,
        xerr=xerr,
        yerr=yerr,
        **std.default.error_bar_def,
        label=rf"$R^2$={goodness}",
        color="forestgreen",
    )

    plt.legend(loc="upper left", markerscale=0.0)

    std.default.plt_pretty(r"$\sqrt{1+(\frac{z}{z_0})^2}$ / o.E.", r"$\omega(z)$ / m")

    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()


if __name__ == "__main__":
    main()
