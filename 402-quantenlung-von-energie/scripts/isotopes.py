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
    params["mu_D"] = p.ev(data[2][0], data[2][1])
    params["mu_H"] = p.ev(data[3][0], data[3][1])
    params["sigma_D"] = data[4]
    params["sigma_H"] = data[5]
    # mu, dmu = params['mu_D'][0], params['mu_D'][1]
    # dmu_angle = np.abs(to_beta(mu) - to_beta(mu + dmu))
    # mu_angle = to_beta(mu)
    # params["mu_D_deg"] = (mu_angle, dmu_angle)

    # mu, dmu = params['mu_H'][0], params['mu_H'][1]
    # dmu_angle = np.abs(to_beta(mu) - to_beta(mu + dmu))
    # mu_angle = to_beta(mu)
    # params["mu_H_deg"] = (mu_angle, dmu_angle)
    return params


def calc_isotope_split(data):
    delta_beta = 0.1 * data["split"] / 300
    beta = data["angle"] + 140 - 180
    delta_lambda = delta_beta * grating_const * np.cos(beta)
    print(delta_lambda.format() + " nm")


def get_wavelength(data):
    d2r = 2 * np.pi / 360
    alpha = d2r * (p.ev(140, 0.5))
    beta = d2r * (p.ev(data["angle"], 0.5) + 140 - 180)
    wavelength = grating_const * (np.sin(alpha) + np.sin((beta)))

    wavelength *= std.unit.nm
    print("lambda = " + wavelength.format())
    return wavelength


def rydberg_from_abs_lambda(data):
    transition = (0.25 - (1 / data["n"])**2)
    wavelength = get_wavelength(data)
    RH = 1 / (wavelength * transition)

    mass_electron = 9.1093837139e-31
    mass_proton = 1.67262192595e-27
    reduced_mass = mass_proton * mass_electron / (mass_electron + mass_proton)

    print("R_H = " + RH.format())
    R_inf = RH * mass_electron / reduced_mass

    R_inf = R_inf
    print("R_inf = " + R_inf.format())
    return R_inf


def delta_lambda_from_cmos(data):
    # delta_beta = to_beta(data["mu_H"]) - to_beta(data["mu_D"])
    d2r = 2 * np.pi / 360
    delta_beta = (0.014 / 300) * (data["mu_H"] - data["mu_D"])
    beta = d2r * (p.ev(data["angle"], 0.5) + 140 - 180)
    delta_lambda = grating_const * delta_beta * np.cos(beta)
    return delta_lambda


def rydberg_from_delta(data):
    delta_lambda = delta_lambda_from_cmos(data)
    transition = (0.25 - (1 / data["n"])**2)

    mass_electron = 9.1093837139e-31
    mass_proton = 1.67262192595e-27
    mass_neutron = 1.67492750056e-27
    reduced_mass_H = mass_proton * mass_electron / (mass_electron + mass_proton)
    reduced_mass_D = (mass_proton + mass_neutron) * mass_electron / (mass_electron + mass_proton + mass_neutron)
    reduced_mass_diff = reduced_mass_D - reduced_mass_H

    R_inf = delta_lambda *(1 / std.unit.nm) * transition * mass_electron / reduced_mass_diff
    print(R_inf.format())
    return R_inf


def h_from_R(R):
    term_a = std.unit.electron_mass * (std.unit.electron_charge ** 4)
    term_b = (std.unit.vacuum_permitivity ** 2) * 8 * std.unit.c * R
    h = (term_a / term_b) ** (1/3)
    return h


def main():
    key = argv[1]
    handle = open(key, "r")
    data = [get_data(line) for line in handle.readlines()]
    handle.close()

    _ = list(map(calc_isotope_split, data))
    R_inf = list(map(rydberg_from_abs_lambda, data))
    delta_lambda = list(map(delta_lambda_from_cmos, data))
    _ = [print("delta lambda = " + x.format() + " nm") for x in delta_lambda]
    # _ = list(map(rydberg_from_delta, data))
    for h in map(h_from_R, R_inf):
        print(h.format())


if __name__ == "__main__":
    main()
