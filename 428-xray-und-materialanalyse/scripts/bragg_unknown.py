from sys import argv

import numpy as np
import std
from matplotlib import pyplot as plt

std.bullshit.ger()


def get_data(file):
    data = np.transpose(
        np.loadtxt(
            file,
            skiprows=1,
            delimiter="\t",
            converters=lambda s: float(s.replace(",", ".")),
        )
    )
    pos, counts = data[0], data[1]
    return pos, counts


def plot_raw(file):
    pos, counts = get_data(file)
    std.default.plt_pretty("x", "Intensität")
    plt.plot(pos, counts)
    plt.show()
    return


def convert_data(file):
    pos, counts = get_data(file)
    # wavelength =
    # conversion in richtige physikalische größen für achsen goes here
    return


def main():
    plot_raw(argv[1])
    return


if __name__ == "__main__":
    main()
