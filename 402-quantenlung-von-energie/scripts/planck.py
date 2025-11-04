#!python3
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from scipy.optimize import curve_fit
import propeller as p


def piecwise_linear(x, d, a, alpha, b):
    mask_a = 1 * (x < d)
    mask_b = 1 * (x >= d)
    beta = (a - alpha) * d + b
    return mask_a * (x * a + b) + mask_b * (x * alpha + beta)


eb_defaults = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


def get_U_0(file):
    data = np.transpose(np.loadtxt(file, delimiter=",", skiprows=1))
    voltage = data[0]  # Volt
    current = (data[1] - data[1][0]) * 1e-10  # Ampere
    y = np.sqrt(current)

    param, cov = curve_fit(piecwise_linear, voltage, y, p0=[-1, 0, 0.1, 0])

    errs = np.sqrt(np.diag(cov))
    delim = p.ev(param[0], errs[0])
    a = p.ev(param[1], errs[1])
    alpha = p.ev(param[2], errs[2])
    b = p.ev(param[3], errs[3])
    beta = (a - alpha) * delim + b
    voltage_zero = - beta / alpha
    return voltage_zero
    # return p.value(voltage_zero), p.error(voltage_zero)


def get_data(string):
    chunks = string.split()
    wavelength = float(chunks[0])
    file = "/".join(argv[1].split("/")[:-1] + [chunks[1]])
    print(file)
    zero_voltage = get_U_0(file)
    return wavelength, zero_voltage


def calc_planck(data):
    speed_of_light = 2.998e8
    freqs = np.array([speed_of_light / wavelength for wavelength, _ in data])
    voltage = - np.array([p.value(res) for _, res in data])
    voltage_err = np.array([p.error(res) for _, res in data])

    params, cov = curve_fit(lambda x, a, b: a * x + b, freqs, voltage, sigma=voltage_err)
    errors = np.sqrt(np.diag(cov))
    xrange = np.linspace(min(freqs) - 0.1 * np.average(freqs), max(freqs) + 0.1 * np.average(freqs), 1000)

    alpha = p.ev(params[0], errors[0])
    beta = p.ev(params[1], errors[1])

    electron_charge = 1.602e-19

    planck_constant = alpha * electron_charge
    work = - beta * electron_charge

    print(f"h = {~planck_constant}")
    print(f"W_A = {~work} J")

    plt.plot(xrange, params[0] * xrange + params[1])
    plt.errorbar(freqs, voltage, voltage_err, xerr=freqs * 0.05, **eb_defaults)
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel("Frequenz / Hz")
    plt.ylabel(r"Grenzspannung / V")
    plt.show()


if __name__ == "__main__":
    key = argv[1]
    handle = open(key, "r")
    data = [get_data(line) for line in handle.readlines()]
    handle.close()

    calc_planck(data)
