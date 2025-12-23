#!python3
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
import scipy
import std
from dataclasses import dataclass


def load(file):
    data = np.transpose(np.loadtxt(file, delimiter="\t", skiprows=5))
    return data[2], data[1]


def make_n_gaussian(n):
    return lambda x, *args: sum([std.gaussian(x, args[i], args[i + 1], args[i + 2]) for i in range(0, 3 * n, 3)])


def fit_multi_gauss(x, y):
    # x, xid = np.unique(x, return_index=True)
    # y = y[xid]

    peak_ids, _ = scipy.signal.find_peaks(y, width=5, prominence=0.05, distance=20)

    if x[peak_ids[-1]] < max(x) - 3.:
        peak_ids = np.append(peak_ids, len(x) - 1)


    peak_x = x[peak_ids]

    delta_x = np.abs(0.4 * np.average(peak_x[1:] - peak_x[:-1]))

    a, µ, sigma = [], [], []

    for i in range(len(peak_ids)):
        mask = np.abs(x - peak_x[i]) < delta_x
        x_slice = x[mask]
        y_slice = y[mask]

        p0=[max(y_slice), x_slice[y_slice == max(y_slice)][0], 0.5 * (x_slice[-1] - x_slice[0])]

        # this handles the case where the last peak lies outside the measured range
        if i == len(peak_ids) - 1:
            p0[1] = µ[-1] + (µ[-1] - µ[-2])
            p0[0] = 1.7 * a[-1]
            # p0 = np.interp()

        # print(p0[1])

        params, _ = std.fit_func(std.gaussian, x_slice, y_slice, p0=p0)

        a.append(params[0])
        µ.append(params[1])
        sigma.append(params[2])

        # plt.scatter(x_slice, y_slice, marker="x")
        # plt.plot(x_slice, std.gaussian(x_slice, *params))
        # std.default.plt_pretty("x", "y")
        # plt.show()

    p0 = []
    # print(f"prelim µ: {µ}")
    for i in range(len(peak_ids)):
        p0.append(a[i])
        p0.append(µ[i])
        p0.append(sigma[i])

    f = make_n_gaussian(len(peak_ids))
    params, (std_devs, r_sq) = std.fit_func(f, x, y, p0=p0)

    return params, std_devs, r_sq


@dataclass
class fit_res:
    µ: np.ndarray
    err_µ: np.ndarray
    sigma: np.ndarray
    err_sigma: np.ndarray
    amp: np.ndarray
    err_amp: np.ndarray


def process_file(file, save=False, show=False):
    U_acc, U_I = load(file)

    params, std_devs, r_sq = fit_multi_gauss(U_acc, U_I)
    µ, err_µ = params[1::3], std_devs[1::3]
    sort_key = np.argsort(µ)
    sigma, err_sigma = params[2::3][sort_key], std_devs[2::3][sort_key]
    amp, err_amp = params[0::3][sort_key], std_devs[0::3][sort_key]
    µ = µ[sort_key]

    if show or save:
        delta_µ = µ[1:] - µ[:-1]
        print(f"fittet µ are : {np.vectorize(lambda x: round(x, 2))(µ)}")
        print(f"delta µ is : {delta_µ}")
        print(f"average delta µ is : {np.sum(delta_µ[:-1]) / len(delta_µ[:-1])}")

        eb_param = std.default.error_bar_def
        plt.errorbar(U_acc, U_I, xerr=0.1, yerr=0.1, **eb_param)

        multi_gauss = make_n_gaussian(len(µ))
        xrange = np.linspace(min(U_acc), max(U_acc), 10000)
        plt.plot(xrange, [multi_gauss(x, *params) for x in xrange], label=f"$R^2 = {round(r_sq, 3)}$")
        for i in range(len(µ)):
            plt.plot(xrange, std.gaussian(xrange,  amp[i], µ[i], sigma[i]), linestyle="dashdot")

        std.default.plt_pretty("Beschleunigungsspannung / V", "Strom / Einheit")
        plt.legend()
        plt.show()

    return fit_res(µ, err_µ, sigma, err_sigma, amp, err_amp)


def write_table(data, to_file):
    pass


def main():
    list_file = std.readfile(argv[1])

    temps = [float(x.split(",")[0]) for x in list_file]
    u_max = [float(x.split(",")[1]) for x in list_file]
    u_2 = [float(x.split(",")[2]) for x in list_file]
    file = ["/".join(argv[1].split("/")[:-1]) + "/" + x.split(",")[3].strip() for x in list_file]

    save = argv[-1] == "save"

    id = [0]
    if "a" in argv[2]:
        id = range(len(file))
    elif "-" in argv[2]:
        id = range(int(argv[2].split("-")[0]), int(argv[2].split("-")[1]) + 1)

    filter = np.ones(len(temps))
    if len(argv) > 3:
        T = np.array(temps)
        U = np.array(u_2)
        filter = eval(argv[3])
        _ = T
        _ = U

    peak_pos, peak_num, all_res = [], [], []
    if not save:
        progress = std.pbar(len(id), msg="running fits ")
        progress.next()

    for i in id:
        if not save:
            progress.next()
        if not filter[i]:
            continue
        # print(f"running for: T = {temps[i]}C; U_max = {u_max[i]}V; U_2 = {u_2[i]}V")
        res = process_file(file[i], save)
        µ = res.µ[:-1]
        peak_ids = np.array(range(len(µ)))
        peak_ids += 0 if µ[0] < 10 else 1
        peak_pos = np.append(peak_pos, µ)
        peak_num = np.append(peak_num, peak_ids)
        all_res.append(res)

    peak_pos = peak_pos[peak_num > 0]
    peak_num = peak_num[peak_num > 0]
    params, (err, r_sq) = std.fit_func(lambda x, a, b: a * x + b, peak_num, peak_pos)

    print(params)
    plt.scatter(peak_num, peak_pos, marker="x")
    xrange = np.linspace(0, max(peak_num) + 0.5)
    plt.plot(xrange, (lambda x, a, b: a * x + b)(xrange, *params), label=f"$R^2 = {r_sq}$")
    std.default.plt_pretty("Nummer des Maximums", "$U_B$ / V")
    plt.show()


if __name__ == "__main__":
    main()
