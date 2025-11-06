#!python3
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv
import std

std.bullshit.this_is_fucking_stupid_no_one_actually_gives_a_fuck()


def parse_tf(filename):
    lines = []
    with open(filename) as handle:
        lines = handle.readlines()

    lines = [x.replace('\ufeffp', '') for x in lines[1:]]
    bins = [int(x.split()[0]) for x in lines]
    values = [float(x.split()[1].replace(",", ".")) for x in lines]
    return np.array(bins), np.array(values)


def gaussian(x, amp, mu, sigma):
    temp_a = (x - mu) ** 2
    temp_b = 2 * (sigma ** 2)
    return amp * np.exp(-temp_a / temp_b)


def double_gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2, const):
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2) + const


def to_beta(p):
    return np.rad2deg(np.arctan((1024 - p) * 0.014 / 300))


def from_beta(beta):
    return 1024 - 300 * np.tan(np.deg2rad(beta))


def mu_label(mu, dmu):
    s = r"$\mu = ("
    dmu_angle = np.abs(to_beta(mu) - to_beta(mu + dmu))
    s += str(round(to_beta(mu), 4))
    s += r"\pm"
    s += str(round(dmu_angle, 4))
    s += r")^\circ$"
    return s


def fwhm_label(mu, sigma):
    s = "FWHM $ = "
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_angle = to_beta(mu + 0.5 * fwhm) - to_beta(mu - 0.5 * fwhm)
    s += str(round(fwhm)) + r" \hat{=} " + str(round(fwhm_angle, 2)) + r"\circ"
    s += "$"
    return s


def main():
    bins, values = parse_tf(argv[1])

    window = 75
    max_bin = bins[values == max(values)]
    relevant = (max_bin - window < bins) & (bins < max_bin + window)

    if not len(argv) > 2:
        plt.bar(bins, values)
        plt.show()
        return

    if len(argv) > 5:
        relevant = (int(argv[4]) < bins) & (bins < int(argv[5]))

    bins = bins[relevant]
    values = values[relevant]

    guesses = [10, 20, int(argv[2]), int(argv[3]), 1, 1, min(values)]
    params, cov = curve_fit(double_gaussian, bins, values, p0=guesses)
    errors = np.sqrt(np.diag(cov))
    chi2 = std.reduced_chi_2(values, double_gaussian(bins, *params), params, sigma=1)
    print(f"reduced chi2 = {chi2}")

    xrange = np.linspace(min(bins), max(bins), 10000)
    gauss_1 = gaussian(xrange, params[0], params[2], params[4])
    gauss_2 = gaussian(xrange, params[1], params[3], params[5])

    plt.bar(bins, values, color="silver")
    chi2label = r"$\chi^2_{red} = " + str(round(chi2, 5)) + "$"
    plt.plot(
             xrange, double_gaussian(xrange, *params),
             color="darkviolet", label=chi2label)
    plt.plot(
             xrange, params[6] + gauss_1, linestyle="--",
             color="deepskyblue", label=mu_label(params[2], errors[2]))
    plt.plot(
             xrange, params[6] + gauss_2, linestyle="--",
             color="coral", label=mu_label(params[3], errors[3]))
    plt.legend()

    std.default.plt_pretty("Pixel / Zahl", "Intensität / arbiträre Einheit")
    deg_axis = plt.gca().secondary_xaxis("top", (to_beta, from_beta))
    deg_axis.set_xlabel("Winkel / Grad")

    if len(argv) > 6:
        plt.savefig(argv[6])
    else:
        plt.show()

    if len(argv) > 7:
        with open(argv[7], "w") as handle:
            s = " ".join(map(str, params)) + "\n"
            s += " ".join(map(str, errors)) + "\n"
            handle.write(s)


if __name__ == "__main__":
    main()
