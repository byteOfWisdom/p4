#!python3
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv
import std

# std.bullshit.this_is_fucking_stupid_no_one_actually_gives_a_fuck()

def get_data(string):
    chunks = string.split()
    angle = float(chunks[0])
    file = "/".join(argv[1].split("/")[:-1] + [chunks[1]])
    print(file)
    params = load(file)
    params["angle"] = angle
    return params


def to_beta(p):
    return np.rad2deg(np.arctan((1024 - p) * 0.014 / 300))


def from_beta(beta):
    return 1024 - 300 * np.tan(np.deg2rad(beta))


def load(file):
    data = np.loadtxt(file)
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
    pass


def main():
    key = argv[1]
    handle = open(key, "r")
    data = [get_data(line) for line in handle.readlines()]
    handle.close()

    calc_isotope_split(data)


if __name__ == "__main__":
    main()
