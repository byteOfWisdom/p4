#!python3
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
import scipy
import std


def load(file):
    data = np.transpose(np.loadtxt(file, delimiter="\t", skiprows=5))
    return data[2], data[1]


def penta_gauss(x, *args):
    args = iter(args)
    g1 = std.gaussian(x, next(args), next(args), next(args))
    g2 = std.gaussian(x, next(args), next(args), next(args))
    g3 = std.gaussian(x, next(args), next(args), next(args))
    g4 = std.gaussian(x, next(args), next(args), next(args))
    g5 = std.gaussian(x, next(args), next(args), next(args))
    # g6 = std.gaussian(x, next(args), next(args), next(args))
    return g1 + g2 + g3 + g4 + g5 #+ g6


def main():
    U_acc, U_I = load(argv[1])
    peaks = U_acc[scipy.signal.find_peaks(U_I, width=5, prominence=0.0005)[0]]

    # peaks = np.append([peaks[0] - 5], peaks)
    n_peaks = len(peaks)
    peak_iter = iter(peaks)
    p0 = np.zeros(n_peaks * 3 + 3)
    for i in range(0, n_peaks * 3, 3):
        p0[i] = i
        p0[i + 1] = next(peak_iter)
        p0[i + 2] = 1.5
    p0[-3] = 6
    p0[-2] = peaks[-1] + 5
    p0[-1] = 1.5

    print(peaks)
    params, cov = scipy.optimize.curve_fit(penta_gauss, U_acc, U_I, p0=p0, maxfev=99999)
    eb_param = std.default.error_bar_def

    plt.errorbar(U_acc, U_I, xerr=0.1, yerr=0.1, **eb_param)
    xrange = np.linspace(min(U_acc), max(U_acc), 10000)
    plt.plot(xrange, [penta_gauss(x, *params) for x in xrange])
    std.default.plt_pretty("Beschleunigungsspannung / V", "Strom / Einheit")
    plt.show()


if __name__ == "__main__":
    main()
