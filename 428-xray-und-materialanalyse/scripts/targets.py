from sys import argv

import numpy as np
import scipy
import std
from matplotlib import pyplot as plt

std.bullshit.ger()


# def get_data_bragg(file):
#     data = np.transpose(
#         np.loadtxt(
#             file,
#             skiprows=1,
#             delimiter="\t",
#             converters=lambda s: float(s.replace(",", ".")),
#         )
#     )
#     pos, counts = data[0], data[1]
#     return pos, counts


def get_data_target(file):
    data = np.transpose(np.loadtxt(file, skiprows=1, delimiter="\t"))
    bins, counts = data[0], data[1]
    return bins, counts


def find_ref_peaks(feznfile: str):
    bins, counts = get_data_target(feznfile)
    no_peaks = 4

    # optionally zero for all counts < threshold
    clean_counts = np.linspace(min(bins), max(bins), len(bins))
    for i in bins:
        # print(int(i))
        value = counts[int(i)]
        # print(value)
        if counts[int(i)] < 50:
            clean_counts[int(i)] = 0

        else:
            clean_counts[int(i)] = counts[int(i)]

    # placeholder section for peak channel location finding
    # can't to find the correct 4 peaks here, change params or switch methods
    peak_ids, _ = scipy.signal.find_peaks(counts, distance=3, width=6, prominence=0.1)
    print("Peaks positions:", peak_ids)

    std.default.plt_pretty("x", "Intensität")
    plt.plot(bins, counts, linewidth=1.2)

    # reference peak intensities
    peak_intensities = []
    for i in range(len(peak_ids)):
        peak_intensities.append(counts[peak_ids[i]])
    print("Peak intensities:", peak_intensities)
    return peak_ids, peak_intensities


def linear(x, a, b):
    return a * x + b


def energy_calibration(feznfile: str):
    peak_ids, peak_intensities = find_ref_peaks(feznfile)

    # peak energies: https://xdb.lbl.gov/xdb.pdf
    fe_alpha = 6403.84  # in eV
    fe_beta = 7057.98
    zn_alpha = 8638.86
    zn_beta = 9572.0

    lit_energies = [fe_alpha, fe_beta, zn_alpha, zn_beta]

    # match peak channel to energy
    peak_energies = peak_intensities
    for peak in range(len(peak_energies)):
        peak_energies[peak] = lit_energies[peak]
    print("Peak energies:", peak_energies)

    fitted, cov = scipy.optimize.curve_fit(linear, peak_ids, peak_energies)
    err = np.sqrt(np.diag(cov))
    print("a:", fitted[0])
    print("b:", fitted[1])  # should be close to 0 (low noise)

    goodness = "TO DO"

    # plot channel against energies
    xrange = np.linspace(0, max(peak_ids) + 10, 500)

    std.default.plt_pretty("Kanal n", "Energie [eV]")
    eb_param = std.default.error_bar_def
    plt.errorbar(
        peak_ids,
        peak_energies,
        xerr=0.1,
        yerr=0.1,
        **eb_param,
        label="gemessene Maxima",
    )  # placeholder errors
    plt.plot(
        xrange,
        linear(xrange, fitted[0], fitted[1]),
        label=f"Anpassunggerade mit $R^2$={goodness}",
    )


def plot_raw(file, mode="target"):
    # if mode == "bragg":
    #     x, y = get_data_bragg(file)
    if mode == "target":
        x, y = get_data_target(file)
    else:
        print("invalid mode")
        return
    std.default.plt_pretty("x", "Intensität")
    eb_param = std.default.error_bar_def
    # placeholder errors
    plt.errorbar(x, y, xerr=0.1, yerr=0, **eb_param)

    # plt.plot(x, y)

    return


def main():
    find_ref_peaks(argv[1])
    # energy_calibration(argv[1])
    # plot_raw(argv[1])
    plt.show()
    plt.legend(loc="best")
    return


if __name__ == "__main__":
    main()
