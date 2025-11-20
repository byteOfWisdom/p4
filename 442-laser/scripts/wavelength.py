from sys import argv

import numpy as np
import scipy as sp
import std
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def get_grating_const(filename):
    file = filename.split("_")
    temp = file[1].split("mm")
    grating_const = int(temp[0])
    return grating_const


def get_data(filename):
    data = np.transpose(np.loadtxt(filename, skiprows=1, delimiter=","))
    rep = data[0]
    order = data[1]
    distance = data[2]
    return rep, order, distance


def main():
    grating_const = get_grating_const(argv[1])
    rep, order, distance = get_data(argv[1])
    if grating_const == 500:
        wall = 13e-2
    elif grating_const == 1000:
        wall = 14e-2
    else:
        sys.exit("you didn't measure this file name?!")
    dist_err = 5e-2

    angle = distance / wall  # this feels like the issue
    angle_term = np.sin(angle)

    wavelength = (grating_const * angle_term) / order
    for i in range(len(wavelength)):
        print(wavelength[i], "nm")


if __name__ == "__main__":
    main()
