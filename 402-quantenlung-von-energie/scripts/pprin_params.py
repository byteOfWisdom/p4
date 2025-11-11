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
    params["amp_D"] = p.ev(data[0][0], data[0][1])
    params["amp_H"] = p.ev(data[1][0], data[1][1])
    params["mu_D"] = p.ev(data[2][0], data[2][1])
    params["mu_H"] = p.ev(data[3][0], data[3][1])
    params["sigma_D"] = p.ev(data[4][0], data[4][1])
    params["sigma_H"] = p.ev(data[5][0], data[5][1])
    # mu, dmu = params['mu_D'][0], params['mu_D'][1]
    # dmu_angle = np.abs(to_beta(mu) - to_beta(mu + dmu))
    # mu_angle = to_beta(mu)
    # params["mu_D_deg"] = (mu_angle, dmu_angle)

    # mu, dmu = params['mu_H'][0], params['mu_H'][1]
    # dmu_angle = np.abs(to_beta(mu) - to_beta(mu + dmu))
    # mu_angle = to_beta(mu)
    # params["mu_H_deg"] = (mu_angle, dmu_angle)
    return params


def pprint_params(data, a=True):
    A_D = data['amp_D']
    A_H = data['amp_H']
    mu_H = data['mu_H']
    mu_D = data['mu_D']
    sigma_H = data['sigma_H']
    sigma_D = data['sigma_D']
    sep = "}$ & $\\num{"
    line1 = sep.join(["messung", mu_H.format(), mu_D.format(), sigma_H.format(), sigma_D.format()])
    line2 = sep.join(["messung", A_H.format(), A_D.format()])

    print((line1 if a else line2) + "}$\\\\")


def main():
    key = argv[1]
    handle = open(key, "r")
    data = [get_data(line) for line in handle.readlines()]
    handle.close()

    for x in data:
        pprint_params(x, False)


if __name__ == "__main__":
    main()
