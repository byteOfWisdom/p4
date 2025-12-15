#!python3
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
import scipy
import std

peaknum = 7


def load(file):
    data = np.transpose(np.loadtxt(file, delimiter="\t", skiprows=5))
    return data[2], data[1]


def penta_gauss(x, *args):
    return sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * peaknum, 3)])


def main():
    U_acc, U_I = load(argv[1])
    # peaks = U_acc[scipy.signal.find_peaks(U_I, width=5, prominence=0.0005)[0]]
    peaks = np.linspace(7, 36, peaknum)
    print(peaks)
    amps = np.linspace(1, 7, peaknum)
    sigmas = np.ones(peaknum) * 1.5

    p0 = np.zeros(3 * peaknum)

    for i in range(0, peaknum):
        p0[3 * i] = amps[i]
        p0[3 * i + 1] = peaks[i]
        p0[3 * i + 2] = sigmas[i]

    params, cov = scipy.optimize.curve_fit(penta_gauss, U_acc, U_I, p0=p0, maxfev=99999)
    eb_param = std.default.error_bar_def

    plt.errorbar(U_acc, U_I, xerr=0.1, yerr=0.1, **eb_param)
    xrange = np.linspace(min(U_acc), max(U_acc), 10000)
    plt.plot(xrange, [penta_gauss(x, *params) for x in xrange])
    std.default.plt_pretty("Beschleunigungsspannung / V", "Strom / Einheit")
    plt.show()


if __name__ == "__main__":
    main()
