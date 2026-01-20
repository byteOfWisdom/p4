from sys import argv

import numpy as np
import scipy
import std
from matplotlib import pyplot as plt


def get_data(file):
    data = np.transpose(
        np.loadtxt(
            file,
            skiprows=1,
            delimiter="\t",
            converters=lambda c: float(c.replace(",", ".")),
        )
    )
    return data[0], data[1]


def convert(file, n=1):
    angles, counts = get_data(file)
    std.default.plt_pretty(r"Winkel $\beta$", "Intensität")
    order = n
    abstand = 564.00e-12 / 2
    wavelength = lambda theta: 2 * abstand * np.sin(np.deg2rad(theta)) / n
    std.default.plt_pretty("Wellenlänge", "Intensität")
    plt.plot(wavelength(angles), counts)
    plt.show()


def main():
    convert(argv[1])
    return


if __name__ == "__main__":
    main()
