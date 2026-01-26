#!python3

from matplotlib import pyplot as plt
import scipy
import numpy as np
from sys import argv
import std
from glob import glob
import iminuit
from iminuit import cost
import numba

peak_guesses = {
    "pb": [105, 139, 165, 196, 119],
    "ti": [45, 60, 69, 71],
    "au": [104, 112, 127, 150, 177],
    "fe": [83, 92, 98, 133],
    "w": [73, 96, 110, 127, 150],
    "ni": [97, 108, 113],
    "zr": [116, 152, 205, 231, 235],
    "cu": [105, 117, 121],
    "fezn": [62, 83, 93, 98, 112, 126],
    "sn": [50, 92, 104, 323],
    "ag": [42, 105, 124, 286, 323],
    "zn": [92, 112, 126, 130],
    "in": [47, 54, 83, 105, 117, 311, 347],
    "unknown1": [],
    "unknown2": [],
    "unknown3": []
}


def load_file(fname):
    element = fname.split("_")[0].split("/")[-1]
    data = np.transpose(np.loadtxt(fname, delimiter="\t", skiprows=1))
    return element, data[0], data[1]


def make_n_gaussian(n):
    return numba.njit(lambda x, *args: sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)]))


def fit_peaks(bin, count, name):
    func = make_n_gaussian(len(peak_guesses[name]))
    p0 = []
    for mu in peak_guesses[name]:
        p0.append(0.75 * count[mu])
        p0.append(mu)
        p0.append(5)
    res, (errors, goodness) = std.fit_func(func, bin, count, y_errors=5, p0=p0, force_cf=True)
    # plt.title(name)
    # std.default.plt_pretty("bin", "countrate")
    # plt.scatter(bin, count, marker="x")
    # xrange = np.linspace(min(bin), max(bin), 10000)
    # plt.plot(xrange, func(xrange, *res), color="green")
    # plt.show()

    return name, lambda x: func(x, *res), res


def compose(ref, unkown):
    return None


def make_comp_func(elements):
    func_list = list(elements.values())
    n = len(func_list)
    return numba.njit(lambda x, *ki: sum([(ki[i] ** 2) * func_list[i](x) for i in range(n)]))


def fit_composition(func, x_values, y_values, y_errors=None, p0=None):
    if std.none(y_errors):
        y_errors = np.var(y_values)
    cost_func = cost.LeastSquares(x_values, y_values, y_errors, func)
    m = iminuit.Minuit(cost_func, *p0)
    m.migrad()
    m.hesse()
    goodness = std.goodness_of_fit(y_values, func(x_values, *m.values))
    return m.values, (m.errors, goodness)


def energy_calibration(name, bin, count, refrence_lines):
    _, _, params = fit_peaks(bin, count, name)
    peaks = params[1::3]
    amps = params[0::3]
    params, _ = std.fit_func(lambda x, a, b: a * x + b, peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], refrence_lines)
    # plt.scatter(peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], refrence_lines)
    # plt.plot(peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], (lambda x, a, b: a * x + b)(peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], *params))
    # plt.show()
    return lambda x: (lambda x, a, b: a * x + b)(x, *params)


def main():
    files = glob((argv[1] + "/" if argv[-1] != "/" else argv[1]) + "*" + "_target.txt")
    elements = filter(lambda x: ("unknown" not in x) and ("fezn" not in x), files)
    unknowns = filter(lambda x: "unknown" in x, files)

    print("running energy calibration")
    _, bin, count = load_file(next(filter(lambda x: "fezn" in x, files)))
    fezn_lines = [6.403484, 7.05798, 8.63886, 9.572] # Kalpha, Kbeta for fe then zn in kev
    energy_scale = energy_calibration("fezn", bin, count, fezn_lines)

    print("fitting refrence spectra")
    known_elements = {}
    for elem, bin, count in map(load_file, elements):
        name, spectrum, _ = fit_peaks(bin, count, elem)
        known_elements[name] = spectrum

    # need chromium for sample 2
    chromium = lambda x: known_elements["fe"](x + 12)
    known_elements["cr"] = chromium

    comp_func = make_comp_func(known_elements)
    p0 = np.zeros(len(known_elements.values()))

    print("determining composition")
    for sample, bin, count in map(load_file, unknowns):
        print(f"running for {sample}")
        # res, _ = std.fit_func(comp_func, bin, count, y_errors=5, p0=p0)
        res, _ = fit_composition(comp_func, bin, count, y_errors=5, p0=p0)
        res = np.abs(res)
        abundance = np.flip(np.argsort(res))
        contained_elements = np.array(list(known_elements.keys()))[abundance]
        print(contained_elements)
        print(res[abundance])
        plt.title(sample)
        std.default.plt_pretty("Energie / keV", "Zählrate")
        plt.scatter(energy_scale(bin), count, marker="x")
        xrange = np.linspace(min(bin), max(bin), 10000)
        plt.plot(energy_scale(xrange), comp_func(xrange, *res), color="green")
        plt.show()


if __name__ == "__main__":
    main()
