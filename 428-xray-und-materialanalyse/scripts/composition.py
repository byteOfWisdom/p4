#!python3

from matplotlib import pyplot as plt
import numpy as np
from sys import argv
import std
from glob import glob
import iminuit
from iminuit import cost


save_flag = argv[2] == "save" if len(argv) > 2 else False

energy_scale = None

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
    "unknown1": [70, 83, 89, 104],
    # "unknown2": [90, 105, 111, 118, 121],
    "unknown2": [105, 111, 118, 121],
    "unknown3": [105, 111]
    # "unknown3": [105, 111, 119]
}


# density over atomic mass
# densities = {
#     "pb": 11.3 / 207.2,
#     "ti": 4.11 / 47.867,
#     "au": 19.3 / 196.97,
#     "fe": 7.87 / 55.845,
#     "w": 19.3 / 183.84,
#     "ni": 8.90 / 58.693,
#     "zr": 6.52 / 91.222,
#     "cu": 8.96 / 63.546,
#     "sn": 7.265 / 118.71,
#     "ag": 10.5 / 107.87,
#     "zn": 7.14 / 65.38,
#     "in": 7.31 / 114.82,
#     "cr": 7.15 / 51.996
# }

densities = {
    "pb": 11.3,
    "ti": 4.11,
    "au": 19.3,
    "fe": 7.87,
    "w": 19.3,
    "ni": 8.90,
    "zr": 6.52,
    "cu": 8.96,
    "sn": 7.265,
    "ag": 10.5,
    "zn": 7.14,
    "in": 7.31,
    "cr": 7.15
}

# conversion_const = 1e-3 / 1.66053906892e-27
conversion_const = 1
for key in densities:
    densities[key] *= conversion_const


def load_file(fname):
    element = fname.split("_")[0].split("/")[-1]
    data = np.transpose(np.loadtxt(fname, delimiter="\t", skiprows=1))
    return element, data[0], data[1]


def make_n_gaussian(n):
    return lambda x, *args: sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


def fit_peaks(bin, count, name):
    func = make_n_gaussian(len(peak_guesses[name]))
    p0 = []
    for mu in peak_guesses[name]:
        p0.append(0.75 * count[mu])
        p0.append(mu)
        p0.append(5)
    res, (errors, goodness) = std.fit_func(func, bin, count, y_errors=5, p0=p0, force_cf=True)
    if save_flag:
        plt.title(name)
        std.default.plt_pretty("bin", "countrate")
        plt.scatter(bin, count, marker="x")
        xrange = np.linspace(min(bin), max(bin), 10000)
        plt.plot(xrange, func(xrange, *res), color="green")
        # plt.show()
        plt.savefig(f"../figs/ref_{name}.pdf")
        plt.cla()

    return name, lambda x: func(x, *res), res


def make_comp_func(elements):
    func_list = list(elements.values())
    n = len(func_list)
    return lambda x, *ki: sum([(ki[i] ** 2) * func_list[i](x) for i in range(n)])


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
    params, _ = std.fit_func(lambda x, a, b: a * x + b, peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], refrence_lines,force_cf=True)
    # plt.scatter(peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], refrence_lines)
    # plt.plot(peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], (lambda x, a, b: a * x + b)(peaks[np.flip(np.argsort(amps))[:len(refrence_lines)]], *params))
    # plt.show()
    return lambda x: (lambda x, a, b: a * x + b)(x, *params)


def mass_fractions(elements, amplitudes):
    density_amplitudes = np.array([amplitudes[i] * densities[elements[i]] for i in range(len(elements))])
    sum_of_parts = np.sum(density_amplitudes)
    res = {}
    for i in range(len(elements)):
        res[elements[i]] = density_amplitudes[i] / sum_of_parts
    return res


def calculate_composition(sample, bin, count, comp_func, known_elements):
    p0 = np.zeros(len(known_elements.values()))
    _, sample_func, _ = fit_peaks(bin, count, sample)
    x = np.linspace(0, 512, 4 * 512)
    y = sample_func(x)
    res, _ = fit_composition(comp_func, x, y, y_errors=5, p0=p0)
    res = np.sqrt(np.abs(res))
    abundance = np.flip(np.argsort(res))
    contained_elements = np.array(list(known_elements.keys()))[abundance]
    print(contained_elements)
    print(res[abundance])
    mfs = mass_fractions(contained_elements, res[abundance])
    for elem in mfs:
        print(f"{elem}: {round(mfs[elem] * 100, 2)}")
    if save_flag:
        plt.title(sample)
        std.default.plt_pretty("Energie / keV", "Zählrate")
        plt.plot(energy_scale(x), y, linestyle="dashed")
        xrange = np.linspace(min(bin), max(bin), 10000)
        plt.plot(energy_scale(xrange), comp_func(xrange, *(res**2)), color="green")
        # plt.show()
        plt.savefig(f"../figs/composition_{sample}.pdf")
        plt.cla()

    

def main():
    files = glob((argv[1] + "/" if argv[-1] != "/" else argv[1]) + "*" + "_target.txt")
    elements = filter(lambda x: ("unknown" not in x) and ("fezn" not in x), files)
    unknowns = filter(lambda x: "unknown" in x, files)

    # for name, x, y in map(load_file, unknowns):
    #     plt.title(name)
    #     plt.plot(x, y)
    #     plt.show()

    print("running energy calibration")
    _, bin, count = load_file(next(filter(lambda x: "fezn" in x, files)))
    fezn_lines = [6.403484, 7.05798, 8.63886, 9.572] # Kalpha, Kbeta for fe then zn in kev
    global energy_scale
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

    print("determining composition")
    for data in map(load_file, unknowns):
        print(f"running for {data[0]}")
        calculate_composition(*data, comp_func, known_elements)


if __name__ == "__main__":
    main()
