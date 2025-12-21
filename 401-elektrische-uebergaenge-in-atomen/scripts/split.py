#!python3
import numpy as np
import std
import scipy
from matplotlib import pyplot as plt
from sys import argv
from dataclasses import dataclass
from uncertainties import ufloat


def calib_curve(cv):
    a = 0.1 # todo: use the real values
    b = 0.1
    c = 0.1
    return a * cv + b * cv ** 2 + c * cv ** 3


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
    cutoff = min(y) + 150  # 150 kinda be heuristic
    secs = sections(y > cutoff)
    x_chunks = [x[a:b] for a, b in secs]
    y_chunks = [y[a:b] for a, b in secs]
    from_middle = np.array([np.average(c) - np.average(x) for c in x_chunks])
    central = np.argsort(np.abs(from_middle))[0]
    order = np.abs(np.arange(0, len(x_chunks)) - central)

    return list(zip(x_chunks, y_chunks, order))
 

def make_n_gaussian(n):
    return lambda x, *args: args[-1] + sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


def diff_find_maxima(y, smoothing=2):
    smooth_grad = np.gradient(np.convolve(y, np.ones(2 * smoothing), mode="same"))

    peaks = []

    before, after = np.zeros(len(y), dtype=np.bool), np.zeros(len(y), dtype=np.bool)
    for i in range(smoothing):
        before[i] = 1
        after[i + smoothing] = 1

    for i in range(smoothing, len(y) - smoothing):
        if np.all(smooth_grad[before] > 0) and np.all(smooth_grad[after] < 0):
            peaks.append(i - 1)
        before = np.roll(before, 1)
        after = np.roll(after, 1)

    return peaks


@dataclass
class peak_descriptor:
    height: ufloat
    sigma: ufloat
    fwhm: ufloat
    position: ufloat
    valid: bool
    order: int


def fit_order(x, y, order):
    peaks = diff_find_maxima(y, smoothing=len(y) // 25)
    # peaks -= len(kernel)
    print(len(peaks), x[peaks])
    
    func = make_n_gaussian(len(peaks))
    # print([y[max(0, peak - 5):peak+5] for peak in peaks])
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
    

def wavelenght(x, m):
    refraction_index = 1.457
    etalon_thickness = 0.004
    distance = 0.145
    alpha = x.nominal_value / distance
    return 2 * etalon_thickness * np.sqrt(refraction_index ** 2 - np.sin(alpha) ** 2)


def energy_split(alpha_pi, alpha_sigma):
    wavelength_pi_sigma = 644e-9
    h_ev = 6.6e-16
    return (std.unit.c * h_ev / wavelength_pi_sigma) * (1 - (wavelenght(alpha_pi, 1) / wavelenght(alpha_sigma, 1)))
    # return wavelength_pi_sigma * (1 - (wavelenght(alpha_pi, 1) / wavelenght(alpha_sigma, 1)))


def main():
    x, y, I = load_file(argv[1], 0.145)
    peaks = []
    for x, y, n in isolate_orders(x, y)[1:-1]:
        peaks += fit_order(x, y, n)
        plt.plot(x, y)
        print(energy_split(peaks[1].position, peaks[0].position))

    zero_pos = list(filter(lambda p: p.order == 0, peaks))[1].position
    print(zero_pos)
    for i in range(len(peaks)):
        peaks[i].position -= zero_pos

    _ = [print(f"{'{:.2uS}'.format(p.position)}, {'{:.2uS}'.format(p.height)}, {wavelenght(p.position, p.order)}") for p in peaks]

    px = np.array([p.position.nominal_value for p in peaks])
    py = [p.height.nominal_value for p in peaks]
    plt.scatter(px + zero_pos.nominal_value, py, marker="x", color="purple")

    std.default.plt_pretty("Position / m", "Intentsität / Beliebige Einheit")
    plt.show()


if __name__ == "__main__":
    main()
