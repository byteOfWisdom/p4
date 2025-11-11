#!python3
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
import std
import propeller as p

# std.bullshit.this_is_fucking_stupid_no_one_actually_gives_a_fuck()

grating_const = p.ev(489, 5)  # nm


def get_data(string):
    chunks = string.split()
    file = "/".join(argv[1].split("/")[:-1] + [chunks[0]])
    # print(file)
    params = load(file)
    params["angle"] = p.ev(float(chunks[1]), 0.5)
    params["split"] = float(chunks[2])
    params["n"] = int(chunks[3])
    return params


def to_beta(p):
    return np.rad2deg(np.arctan((1024 - p) * 0.014 / 300))


def from_beta(beta):
    return 1024 - 300 * np.tan(np.deg2rad(beta))


def load(file):
    data = np.transpose(np.loadtxt(file))
    params = {}
    params["amp_D"] = data[0]
    params["amp_H"] = data[1]
    params["mu_D"] = p.ev(data[2][0], data[2][1])
    params["mu_H"] = p.ev(data[3][0], data[3][1])
    params["sigma_D"] = p.ev(data[4][0], data[4][1])
    params["sigma_H"] = p.ev(data[5][0], data[5][1])
    return params


def lw_lambda(data, lw):
    d2r = 2 * np.pi / 360
    delta_beta = 0.1 * lw / 300
    beta = data["angle"] + 140 - 180
    beta *= d2r
    delta_lambda = delta_beta * grating_const * np.cos(beta)
    # print(~np.cos(beta))
    return delta_lambda


def get_wavelength(data):
    d2r = 2 * np.pi / 360
    alpha = d2r * (p.ev(140, 0.5))
    beta = d2r * (p.ev(data["angle"], 0.5) + 140 - 180)
    wavelength = grating_const * (np.sin(alpha) + np.sin((beta)))

    wavelength *= std.unit.nm
    # print("lambda = " + wavelength.format())
    return wavelength


def lw_nu(data, lw):
    d2r = 2 * np.pi / 360
    delta_beta = 0.014 * lw / 300
    beta = data["angle"] + 140 - 180
    beta *= d2r
    delta_lambda = delta_beta * grating_const * np.cos(beta)
    # print(~np.cos(beta))
    return delta_lambda


def line_width(data):
    fwhm = np.sqrt(8 * np.log(2))
    measured_lw_H = data["sigma_H"] * fwhm
    measured_lw_D = data["sigma_D"] * fwhm

    lw_H_lambda = lw_lambda(data, measured_lw_H)
    lw_D_lambda = lw_lambda(data, measured_lw_D)

    print("H width (nm): " + lw_H_lambda.format())
    print("D width (nm): " + lw_D_lambda.format())

    print("theoretical line width:")
    delta_nu_natural = 1

    doppler_lw = 2 * nu_0 * np.sqrt(2 * std.unit.boltzmann_const * np.log(2) / hydrogen_mass)

    print("doppler lw: " + to_lambda_lw(doppler_lw).format())


def to_lambda_lw(nu, delta_nu):
    return std.unit.c * (1 / (nu - 0.5 * delta_nu) - 1 / (nu + 0.5 * delta_nu))
    return - std.unit.c * delta_nu / (nu ** 2)


def main():
    key = argv[1]
    handle = open(key, "r")
    data = [get_data(line) for line in handle.readlines()]
    handle.close()

    for point in data:
        line_width(point)


if __name__ == "__main__":
    main()
