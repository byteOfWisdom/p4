#!python3
import scipy
import numpy as np
from matplotlib import pyplot as plt
import std
import propeller as p
from sys import argv


def main():
    data = np.transpose(np.loadtxt(argv[1], delimiter=",", skiprows=11))
    lower_id = int(argv[2]) if len(argv) > 3 else 0
    upper_id = int(argv[3]) if len(argv) > 3 else len(data[0])
    time_const = 1
    time = data[0] * time_const
    voltage = data[1] * 1e-3

    time = time[lower_id:upper_id]
    voltage = voltage[lower_id:upper_id]

    peaks, _ = scipy.signal.find_peaks(voltage, width=50, distance=250, prominence=0.01)

    # print(peaks)

    dists = peaks[1:] - peaks[:-1]
    seperate = list(dists).index(max(dists))
    batch_1 = peaks[:seperate + 1]
    batch_2 = peaks[seperate + 1:]

    plt.plot(time, voltage)
    # for x in peaks:
    # plt.vlines(time[peaks], min(voltage), max(voltage), color="green")
    plt.vlines(time[batch_1], min(voltage), max(voltage), color="green")
    plt.vlines(time[batch_2], min(voltage), max(voltage), color="red")
    std.default.plt_pretty("time", "voltage")
    plt.show()

    external_spacing = batch_2 - batch_1
    external_spacing = np.average(external_spacing)
    internal_spacing = np.append(batch_1[1:] - batch_1[:-1], batch_2[1:] - batch_2[:-1])

    ids = np.array(list(range(len(internal_spacing))))
    # params, meta = std.fit_func(lambda x, a, b: a * x + b, ids, internal_spacing, p0=[0, 1])
    # meta.pprint()
    res = scipy.stats.linregress(ids, internal_spacing)
    params = [res.slope, res.intercept]
    b = p.ev(res.intercept, res.intercept_stderr)
    print(f"a = {res.slope} +- {res.stderr}")
    print(f"b = {res.intercept} +- {res.intercept_stderr}")

    rel_spacing = b / external_spacing
    print(rel_spacing.format())

    laser_cavity = p.ev(51.3e-2, 1e-2)

    laser_mode_spacing = std.unit.c / (2 * laser_cavity)
    print(f"assuming LEM modes, external res len is {(laser_cavity * rel_spacing / 2).format()}")
    external_assumed = p.ev(4e-2, 2e-2)
    print(f"with {external_assumed.format()} external res len, relative mode spacing is {(2 * external_assumed / laser_cavity).format()}")

    plt.errorbar(ids, internal_spacing, fmt="x")
    plt.plot(ids, (lambda x: (x - x) + external_spacing)(ids))
    plt.plot(ids, params[0] * ids + params[1])
    std.default.plt_pretty("index", "relativer modenabstand")
    plt.show()
    


if __name__ == "__main__":
    main()
