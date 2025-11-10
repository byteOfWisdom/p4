#!python3
import numpy as np
from scipy.optimize import curve_fit
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
    params["angle"] = float(chunks[1])
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
    params["mu_D"] = data[2]
    params["mu_H"] = data[3]
    params["sigma_D"] = data[4]
    params["sigma_H"] = data[5]
    mu, dmu = params['mu_D'][0], params['mu_D'][1]
    dmu_angle = np.abs(to_beta(mu) - to_beta(mu + dmu))
    mu_angle = to_beta(mu)
    params["mu_D_deg"] = (mu_angle, dmu_angle)

    mu, dmu = params['mu_H'][0], params['mu_H'][1]
    dmu_angle = np.abs(to_beta(mu) - to_beta(mu + dmu))
    mu_angle = to_beta(mu)
    params["mu_H_deg"] = (mu_angle, dmu_angle)
    return params


def calc_isotope_split(data):
    delta_beta = 0.1 * data["split"] / 300
    beta = data["angle"] + 140 - 180
    delta_lambda = delta_beta * grating_const * np.cos(beta)
    print(delta_lambda.format() + " nm")


def get_wavelength(data):
    alpha = np.deg2rad(140)
    beta = np.deg2rad(data["angle"] + 140 - 180)
    wavelength = ~grating_const * (np.sin(alpha) + np.sin((beta)))

    wavelength *= std.unit.nm
    print(wavelength)
    return wavelength


def rydberg_from_abs_lambda(data):
    transition = (0.25 - (1 / data["n"])**2)
    wavelength = get_wavelength(data)
    RH = 1 / (wavelength * transition)

    mass_electron = 9.1093837139e-31
    mass_proton = 1.67262192595e-27
    reduced_mass = mass_proton * mass_electron / (mass_electron + mass_proton)

    print(RH)
    R_inf = RH * mass_electron / reduced_mass

    R_inf = R_inf
    print(R_inf)
    return R_inf


def main():
    key = argv[1]
    handle = open(key, "r")
    data = [get_data(line) for line in handle.readlines()]
    handle.close()

    _ = list(map(calc_isotope_split, data))
    _ = list(map(rydberg_from_abs_lambda, data))


if __name__ == "__main__":
    main()
