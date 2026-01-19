# this is pretty much not necessary and handled by the splits.py file

from sys import argv

import calibration
import numpy as np
import scipy
import propeller as p
import std
from matplotlib import pyplot as plt


def get_currents(file):
    data = np.transpose(np.loadtxt(file, delimiter=",", skiprows=1))
    indices = data[0]
    currents = p.ev(data[1], 0.2)
    return indices, currents

def calibration_func(x):
    return calibration.field_function(x)

def get_fields():
    xrange = np.linspace(-10,10,100)
    indices, currents = get_currents(argv[1])
    fields = calibration_func(currents)
    field_vals, field_errs = p.ve(fields)
    return field_vals, field_errs

def get_data(file):
    data = np.transpose(np.loadtxt(file, delimiter=";", skiprows=5))
    pixels = data[0]
    intensities = data[1]
    pixel_num = max(pixels)
    scaled_pixel = pixels / pixel_num

    return scaled_pixel, pixels, intensities

def get_index(file):
    _, file_index = file.split("_")
    file_index, _ = file_index.split(".")
    file_index = int(file_index)
    return file_index

def get_field(file):
    field_vals, field_errs = get_fields()
    field, field_err = round(field_vals[get_index(file)],3), round(field_errs[get_index(file)],3)
    return field, field_err

def plot_all():
    std.bullshit.ger()
    eb_param = std.default.error_bar_def
    if len(argv) < 3:
        std.default.plt_pretty("rel. Position / %", "Intensität / o.E.")
        for i in range(14,27):
            file = f"data/zeeman/a203/ZeemanX_0{i}.txt"
            scaled_pixel, pixels, intensities = get_data(file)
            plt.scatter(scaled_pixel, intensities, marker="x", linewidths=0.3, color="crimson")
        for i in range(14,27):
            file = f"data/zeeman/a203/ZeemanY_0{i}.txt"
            scaled_pixel, pixels, intensities = get_data(file)
            plt.scatter(scaled_pixel, intensities, marker="x", linewidths=0.3, color="darkblue")
        plt.savefig(f"figs/zeeman_pixel_relativ.pdf")
    if len(argv) == 3:
        axis = argv[2].upper()
        std.default.plt_pretty("Position / Pixel ", "Intensität / o.E.")
        for i in range(14,27):
            file = f"data/zeeman/a203/Zeeman{axis}_0{i}.txt"
            scaled_pixel, pixels, intensities = get_data(file)
            plt.scatter(pixels, intensities, marker="x", linewidths=0.5)
        plt.savefig(f"figs/zeeman_pixel_{axis}.pdf")

    #plt.show()
    #

def inner_rings():
    # TODO

def main():
    plot_all()


if __name__ == "__main__":
    main()
