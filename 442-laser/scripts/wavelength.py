from sys import argv, exit

import numpy as np
import scipy as sp
import std
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def get_grating(filename):
    file = filename.split("_")
    temp = file[1].split("mm")
    grating = int(temp[0])
    grating_const = 1e-3 / grating
    if grating == 500:
        wall = 13e-2
    elif grating == 1000:
        wall = 14e-2
    else:
        wall = 0
        sys.exit("you didn't measure this file name?!")

    return grating_const, wall


def get_data(filename):
    data = np.transpose(np.loadtxt(filename, skiprows=1, delimiter=","))
    rep = data[0]
    order = data[1]
    distance = data[2]
    return rep, order, distance


def main():
    grating_const, wall = get_grating(argv[1])
    rep, order, distance = get_data(argv[1])

    print("g:", grating_const, "m")
    dist_err = 5e-2

    angle = (distance / 2) * wall  # this feels like the issue

    print("angles:", angle)

    angle_term = np.cos(angle)

    wavelength = grating_const * angle_term / order
    for i in range(len(wavelength)):
        print(
            "messung:",
            int(rep[i]),
            "order:",
            int(order[i]),
            "lambda:",
            round(wavelength[i] * 1e9, 3),
            "nm",
        )
        # i mean they're at least consistent between messungen??? idk anymore


if __name__ == "__main__":
    main()
