#!python3
import numpy as np
import std
from matplotlib import pyplot as plt
from sys import argv
from dataclasses import dataclass
from uncertainties import ufloat
from glob import glob
from functools import reduce


def calib_curve(current):
    a = ufloat(-0.00032017, 3.79889947e-06)
    b = ufloat(0.00014896, 1.51311471e-05)
    c = ufloat(0.093403, 2.61723881e-04)
    d = ufloat(0.0011227, 7.53829814e-04)
    cubic = (a * (current**3)) + (b * (current**2)) + (c * current) + d
    return cubic


def load_file(fname, lens_dist):
    data = np.transpose(np.loadtxt(fname, delimiter=";", skiprows=5))
    current = float(std.readfile(fname)[0].split()[2])

    pixel_spacing = 9.6e-6
    x = data[0] * pixel_spacing
    return x, data[1], current


def sections(arr, min_lenght=5):
    secs, start = [], 0
    active = False
    for i in range(len(arr)):
        if arr[i] and not active:
            active = True
            start = i
        if not arr[i] and active:
            active = False
            if i - start > min_lenght:
                secs.append((start, i))
    return secs


def isolate_orders(x, y):
    cutoff = min(y) + 200  # 150 kinda be heuristic
    secs = sections(y > cutoff)
    x_chunks = [x[a:b] for a, b in secs]
    y_chunks = [y[a:b] for a, b in secs]
    from_middle = np.array([np.average(c) - np.average(x) for c in x_chunks])
    central = np.argsort(np.abs(from_middle))[0]
    order = np.abs(np.arange(0, len(x_chunks)) - central)

    return list(zip(x_chunks, y_chunks, order))


def make_n_gaussian(n):
    return lambda x, *args: args[-1] + sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


@dataclass
class peak_descriptor:
    height: ufloat
    sigma: ufloat
    fwhm: ufloat
    position: ufloat
    valid: bool
    order: int


def fit_order(x, y, order):
    peaks = std.diff_find_maxima(y, smoothing=len(y) // 30)
    # print(len(peaks), x[peaks])

    func = make_n_gaussian(len(peaks))
    amp_initial = [max(y[max(0, peak - 5):peak + 5]) for peak in peaks]
    µ_initial = [(x[max(0, peak - 5):peak + 5])[y[max(0, peak - 5):peak + 5] == max(y[max(0, peak - 5):peak + 5])][0] for peak in peaks]
    sigma_initial = (max(x) - min(x)) / 10
    p0 = []
    for i in range(len(peaks)):
        p0.append(amp_initial[i])
        p0.append(µ_initial[i])
        p0.append(sigma_initial)
    p0.append(min(y))

    params, (errors, goodness) = std.fit_func(func, x, y, p0=p0, force_cf=True)

    res = []
    for i in range(len(peaks)):
        valid = (params[3 * i] + params[-1] < 1.1 * max(y)) and (min(x) < params[3 * i + 1] < max(x))
        peak = peak_descriptor(
                               height=ufloat(params[3 * i], errors[3 * i]) + ufloat(params[-1], errors[-1]),
                               sigma=ufloat(params[3 * i + 2], errors[3 * i + 2]),
                               fwhm=ufloat(params[3 * i + 2], errors[3 * i + 2]) * 2.355,
                               position=ufloat(params[3 * i + 1], errors[3 * i + 1]),
                               valid=valid, order=order)
        if peak.valid:
            res.append(peak)
        else:
            print("rececting implausible fit")

    return res


def etalon_term(x):
    refraction_index = 1.457
    # etalon_thickness = 4e-3
    distance = 0.145
    return np.sqrt(refraction_index ** 2 - np.sin(x.nominal_value / distance) ** 2)


def energy_split(x_pi, x_sigma):
    wavelength_pi_sigma = 643.8e-9
    h_ev = std.unit.planck_const_eV
    return (h_ev * std.unit.c / wavelength_pi_sigma) * (1 - (etalon_term(x_pi) /etalon_term(x_sigma)))


def process_file(fname, preview=False):
    x, y, I = load_file(fname, 0.145)
    if preview:
        plt.ylim(min(y) - 100, max(y) + 100)
    b_field = calib_curve(I)
    peaks = []
    orders = isolate_orders(x, y)[1:-1]
    middle = np.average(next(filter(lambda o: o[2] == 0, orders))[0])
    x -= middle
    peaks_of_order = []
    for x, y, n in isolate_orders(x, y)[1:-1]:
        fit_res = fit_order(x, y, n)
        peaks += fit_res
        peaks_of_order.append((n, fit_res))
        plt.plot(x, y)

    energies = []

    for order, ps in peaks_of_order:
        if order not in [1, 2, 3, 4] or len(ps) == 1:
            continue
        if len(ps) == 3:
            x_pi = ps[1].position
            x_sigma = ps[0].position
            x_sigma_2 = ps[2].position
        else:
            x_sigma = ps[0].position
            x_sigma_2 = ps[1].position
            x_pi = 0.5 * (x_sigma + x_sigma_2)
        energies.append(energy_split(x_pi, x_sigma))
        energies.append(energy_split(x_pi, x_sigma_2))

    if preview:
        px = np.array([p.position.nominal_value for p in peaks])
        py = [p.height.nominal_value for p in peaks]
        plt.scatter(px, py, marker="x", color="purple")

        plt.title(fname)
        std.default.plt_pretty("Position / m", "Intentsität / Beliebige Einheit")
        plt.show()

    return reduce(lambda a, b: a + b, [p for _, p in peaks_of_order], []), energies


def main():
    # 14 - 26, 32, 31? 33?
    for i in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 32, 31, 33]:
        try:
            f = argv[1] + f"ZeemanX_0{i}.txt"
            process_file(f, preview=True)
            print(f"parsed {i} X")
        except Exception as e:
            print(e)
        try:
            f = argv[1] + f"ZeemanY_0{i}.txt"
            process_file(f, preview=True)
            print(f"parsed {i} Y")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
