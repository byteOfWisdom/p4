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
    angle = data[0] * pixel_spacing / lens_dist
    return data[0], data[1], current


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
    return list(zip(x_chunks, y_chunks))
 

def make_n_gaussian(n):
    return lambda x, *args: args[-1] + sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


def diff_find_maxima(y, smoothing=2):
    # smooth_grad = np.roll(0.2 * np.convolve(np.gradient(y), np.ones(2 * smoothing), mode="same"), 0)
    smooth_grad = np.gradient(np.convolve(y, np.ones(2 * smoothing), mode="same"))
    # plt.cla()
    # plt.plot(np.gradient(y))
    # plt.plot(smooth_grad)
    # plt.plot(y)
    # plt.show()

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


def fit_order(x, y):
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
                               valid=valid)
        if peak.valid:
            res.append(peak)
        else:
            print("rececting implausible fit")

    # _ = [print(r) for r in res]
    # print(params)
    # plt.cla()
    # plt.plot(x, y)
    # xrange = np.linspace(min(x), max(x), 1000)
    # plt.plot(xrange, func(xrange, *params))
    # plt.plot(xrange, std.gaussian(xrange, *params[0:3]) + params[-1], linestyle="dotted")
    # plt.plot(xrange, std.gaussian(xrange, *params[3:6]) + params[-1], linestyle="dotted")
    # plt.plot(xrange, std.gaussian(xrange, *params[6:9]) + params[-1], linestyle="dotted")
    # plt.scatter([p.position.nominal_value for p in res], [p.height.nominal_value for p in res], marker="x")
    # plt.show()

    return res
    

def main():
    x, y, I = load_file(argv[1], 0.145)
    peaks = []
    for x, y in isolate_orders(x, y)[1:-1]:
        peaks += fit_order(x, y)
        plt.plot(x, y)

    _ = [print(f"{'{:.2uS}'.format(p.position)}, {'{:.2uS}'.format(p.height)}") for p in peaks]

    px = [p.position.nominal_value for p in peaks]
    py = [p.height.nominal_value for p in peaks]
    plt.scatter(px, py, marker="x", color="purple")

    std.default.plt_pretty("Winkel / rad", "Intentsität / Beliebige Einheit")
    plt.show()


if __name__ == "__main__":
    main()
