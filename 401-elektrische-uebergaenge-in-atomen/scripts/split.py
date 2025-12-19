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


def load_file(fname):
    data = np.transpose(np.loadtxt(fname, delimiter=";", skiprows=5))
    current = float(std.readfile(fname)[0].split()[2])
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
    return lambda x, *args: args[-1] + sum([std.gaussian(x, np.abs(args[i]), args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


@dataclass
class peak_descriptor:
    height: ufloat
    sigma: ufloat
    fwhm: ufloat
    position: ufloat


def fit_order(x, y):
    # plt.cla()
    # plt.plot(x, y)
    # plt.show()
    kernel = np.ones(int(len(x) / 10))
    peaks, _ = scipy.signal.find_peaks(scipy.signal.convolve(y, kernel), width=5, height=(min(y) + 0.3 * (max(y) - min(y))))
    peaks -= len(kernel)
    print(len(peaks), x[peaks])
    func = make_n_gaussian(len(peaks))
    print([y[max(0, peak-5):peak+5] for peak in peaks])
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
        peak = peak_descriptor(
                               height=ufloat(params[3 * i], errors[3 * i]) + ufloat(params[-1], errors[-1]),
                               sigma=ufloat(params[3 * i + 2], errors[3 * i + 2]),
                               fwhm=ufloat(params[3 * i + 2], errors[3 * i + 2]) * 2.355,
                               position=ufloat(params[3 * i + 1], errors[3 * i + 1]))
        res.append(peak)

    # _ = [print(r) for r in res]
    # plt.cla()
    # plt.plot(x, y)
    # xrange = np.linspace(min(x), max(x), 1000)
    # plt.plot(xrange, func(xrange, *params))
    # plt.scatter([p.position.nominal_value for p in res], [p.height.nominal_value for p in res], marker="x")
    # plt.show()

    return res
    

def main():
    x, y, I = load_file(argv[1])
    peaks = []
    for x, y in isolate_orders(x, y)[1:-1]:
        peaks += fit_order(x, y)
        plt.plot(x, y)

    _ = [print(f"{'{:.2uS}'.format(p.position)}, {'{:.2uS}'.format(p.height)}") for p in peaks]

    px = [p.position.nominal_value for p in peaks]
    py = [p.height.nominal_value for p in peaks]
    plt.scatter(px, py, marker="x", color="purple")

    std.default.plt_pretty("Ort", "Intentsität")
    plt.show()


if __name__ == "__main__":
    main()
