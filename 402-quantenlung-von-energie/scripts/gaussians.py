#!python3
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv


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
    plt.bar(bins, values, color="silver")
    xrange = np.linspace(min(bins), max(bins), 10000)
    plt.plot(xrange, double_gaussian(xrange, *params), color="darkviolet")
    gauss_1 = gaussian(xrange, params[0], params[2], params[4])
    gauss_2 = gaussian(xrange, params[1], params[3], params[5])
    plt.plot(xrange, params[6] + gauss_1, linestyle="--", color="deepskyblue")
    plt.plot(xrange, params[6] + gauss_2, linestyle="--", color="coral")
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel("Pixel")
    plt.ylabel("Licht")
    plt.show()


if __name__ == "__main__":
    main()
