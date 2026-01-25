#!python3

from matplotlib import pyplot as plt
import scipy
import numpy as np
from sys import argv
import std
from glob import glob

peak_guesses = {
    ""
}


def load_file(fname):
    element = fname.split("_")[0]
    data = np.transpose(np.loadtxt(fname, delimiter="\t", skiprows=1))
    return element, data[0], data[1]


def fit_peaks(bin, count):
    std.default.plt_pretty("bin", "countrate")
    plt.plot(bin, np.convolve(count,  np.ones(1), mode="same"))
    plt.show()


def main():
    files = glob((argv[1] + "/" if argv[-1] != "/" else argv[1]) + "*" + "_target.txt")
    print(argv[1] + "*" + "_target.txt")
    elements = filter(lambda x: "unknown" not in x, files)
    unknowns = filter(lambda x: "unknown" in x, files)

    refrence_data = map(load_file, elements)
    for elem, bin, count in refrence_data:
        plt.title(elem)
        fit_peaks(bin, count)




if __name__ == "__main__":
    main()
