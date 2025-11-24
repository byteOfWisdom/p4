#!python3
import scipy
import numpy as np
from matplotlib import pyplot as plt
import std
import propeller as p
from sys import argv


def load_slice(fname, lower, upper):
    data = np.transpose(np.loadtxt(fname, delimiter=",", skiprows=11))

    time_const = 1
    time = data[0] * time_const
    voltage = data[1] * 1e-3

    time = time[lower:upper]
    voltage = voltage[lower:upper]
    return time, voltage


def get_peaks(voltage):
    peaks, _ = scipy.signal.find_peaks(voltage, width=30, distance=250, prominence=0.0075)
    # peaks, _ = scipy.signal.find_peaks(voltage, width=20, distance=250)
    return peaks


def batches(peaks):
    dists = peaks[1:] - peaks[:-1]
    seperate = list(dists).index(max(dists))
    batch_1 = peaks[:seperate + 1]
    batch_2 = peaks[seperate + 1:]
    return batch_1, batch_2


def plot_measurement(time, peaks, voltage, fmt, color):
    plt.plot(time, voltage, linestyle=fmt, color=next(color))
    batch_1, batch_2 = batches(peaks)
    plt.vlines(time[batch_1], min(voltage), max(voltage), color=next(color))
    plt.vlines(time[batch_2], min(voltage), max(voltage), color=next(color))


def main():
    files = [
        ("../data/20251118_225116.csv", 36000, 66000),
        ("../data/20251118_224854.csv", 32000, 63000),
        ("../data/20251118_225143.csv", 36000, 66000)
    ]

    ext_spacings = np.array([])
    internal_spacings = np.array([], dtype=np.float64)

    fmts = iter(("solid", "solid", "solid"))
    colors = iter((
        iter(("tab:blue", "tab:red", "tab:green")),
        iter(("tab:blue", "tab:red", "tab:green")),
        iter(("tab:blue", "tab:red", "tab:green")),
    ))

    for arg in argv[1:]:
        id = int(arg)
        time, voltage = load_slice(*files[id])
        peaks = get_peaks(voltage)
    
        batch_1, batch_2 = batches(peaks)

        external_spacing = batch_2 - batch_1
        external_spacing = np.average(external_spacing)
        internal_spacing = np.append(batch_1[1:] - batch_1[:-1], batch_2[1:] - batch_2[:-1])

        ext_spacings = np.append(ext_spacings, external_spacing)
        internal_spacings = np.append(internal_spacings, internal_spacing)
        plot_measurement(time, peaks, voltage, next(fmts), next(colors))
        std.default.plt_pretty("Index", "Spannung / mV")
        # plt.show()
        plt.savefig(f"../figs/analysator_spectrum_{id}.pdf")
        plt.cla()

    ids = np.array(list(range(len(internal_spacings))))
    # params, meta = std.fit_func(lambda x, a, b: a * x + b, ids, internal_spacing, p0=[0, 1])
    # meta.pprint()
    res = scipy.stats.linregress(ids, internal_spacings)
    params = [res.slope, res.intercept]
    b = p.ev(res.intercept, res.intercept_stderr)
    print(f"a = {res.slope} +- {res.stderr}")
    print(f"b = {res.intercept} +- {res.intercept_stderr}")
    print(f"ext index spacing = {np.average(ext_spacings)}")

    q = b / np.average(ext_spacings)
    print("q = ", q.format())

    # laser_cavity = p.ev(51.3e-2, 1e-2)
    external_assumed = p.ev(5e-2, 1.5e-2)
    # external_assumed = p.ev(1.5e-2, 1.5e-2)

    # laser_mode_spacing = std.unit.c / (2 * laser_cavity)
    external_mode_spacing = std.unit.c / (4 * external_assumed)
    # print(f"assuming LEM modes, external res len is {(laser_cavity * rel_spacing / 2).format()}")
    # print(f"with {external_assumed.format()} external res len, relative mode spacing is {(2 * external_assumed / laser_cavity).format()}")

    print(f"external mode spacing = {external_mode_spacing.format()}")
    print(f"laser mode spacing = {(q * external_mode_spacing).format()}")

    plt.errorbar(ids, internal_spacings, fmt="x")
    plt.plot(ids, (lambda x: (x - x) + np.average(ext_spacings))(ids))
    plt.plot(ids, params[0] * ids + params[1])
    std.default.plt_pretty("Index", "relativer Modenabstand")
    # plt.show()
    plt.savefig("../figs/modespacing.pdf")
    plt.cla()


if __name__ == "__main__":
    main()
