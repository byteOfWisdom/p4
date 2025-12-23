#!python3
import numpy as np
import std
from matplotlib import pyplot as plt
from sys import argv
from dataclasses import dataclass
import propeller as p
from functools import reduce


def calib_curve(current):
    # a = ufloat(-0.00032017, 3.79889947e-06)
    # b = ufloat(0.00014896, 1.51311471e-05)
    # c = ufloat(0.093403, 2.61723881e-04)
    # d = ufloat(0.0011227, 7.53829814e-04)
    a = p.ev(-0.00032017, 3.79889947e-06)
    b = p.ev(0.00014896, 1.51311471e-05)
    c = p.ev(0.093403, 2.61723881e-04)
    d = p.ev(0.0011227, 7.53829814e-04)
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
    cutoff = min(y) + 175  # 150 kinda be heuristic
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
    height: p.ErrVal
    sigma: p.ErrVal
    fwhm: p.ErrVal
    position: p.ErrVal
    valid: bool
    order: int
    goodness: float


def fit_order(x, y, order, pvalue=0.05):
    if order == 0:
        return []
    peaks = std.diff_find_maxima(y, smoothing=len(y) // 25)
    peaks = std.find_mu(y, 3, smoothing=len(y) // 25)
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

    params, (errors, goodness) = std.fit_func(func, x, y, p0=p0, force_cf=False)

    xrange = np.linspace(min(x), max(x), 1000)
    plt.plot(xrange, np.vectorize(func)(xrange, *params), linestyle="dotted")

    res = []
    for i in range(len(peaks)):
        valid = (params[3 * i] + params[-1] < 1.1 * max(y)) and (min(x) < params[3 * i + 1] < max(x))
        valid = valid and np.abs(1 - goodness) < pvalue
        peak = peak_descriptor(
                               height=p.ev(params[3 * i], errors[3 * i]) + p.ev(params[-1], errors[-1]),
                               sigma=p.ev(params[3 * i + 2], errors[3 * i + 2]),
                               fwhm=p.ev(params[3 * i + 2], errors[3 * i + 2]) * 2.355,
                               position=p.ev(params[3 * i + 1], errors[3 * i + 1]),
                               valid=valid, order=order, goodness=goodness)
        # if peak.valid:
        res.append(peak)
        # else:
            # print("rececting implausible fit")
            # print(peak)

    return res


def etalon_term(x):
    refractive_index = 1.457
    distance = p.ev(0.15, 0.01)
    return np.sqrt(refractive_index ** 2 - np.sin((x / distance)) ** 2)


def energy_split(x_pi, x_sigma):
    wavelength_pi_sigma = 643.847e-9
    h_ev = std.unit.planck_const_eV
    return (h_ev * std.unit.c / wavelength_pi_sigma) * (1 - (etalon_term(x_pi) /etalon_term(x_sigma)))


def process_file(fname, pvalue=0.05, preview=False, save=True):
    x, y, current = load_file(fname, 0.145)
    if preview or save:
        plt.ylim(min(y) - 100, max(y) + 100)
    b_field = calib_curve(current)
    peaks = []
    orders = isolate_orders(x, y)[1:-1]
    middle = np.average(next(filter(lambda o: o[2] == 0, orders))[0])
    # print(middle)
    x -= middle
    peaks_of_order = []
    for x, y, n in isolate_orders(x, y)[1:-1]:
        fit_res = fit_order(x, y, n, pvalue=pvalue)
        peaks += fit_res
        peaks_of_order.append((n, fit_res))
        if preview or save:
            plt.plot(x, y, linewidth=0.4)

    energies = []

    expected_peak_count = round(np.average([len(pc) for _, pc in peaks_of_order]))
    # print(expected_peak_count)
    for order, ps in peaks_of_order:
        if np.any(list(map(lambda x: not x.valid, ps))):
            continue
        if order not in [1, 2, 3, 4, 5, 6] or len(ps) == 1:
            continue
        if len(ps) != expected_peak_count:
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

    if preview or save:
        px = np.array([~p.position for p in peaks])
        py = [~p.height for p in peaks]
        plt.scatter(px, py, marker="x", color="purple")

        plt.title(fname)
        std.default.plt_pretty("Position / m", "Intentsität / Beliebige Einheit")
        dir = "/".join(argv[0].split("/")[:-1]) + "/../figs/"
        if preview:
            plt.show()
        if save:
            # print(dir + fname.split("/")[-1][:-4] + ".pdf")
            plt.savefig(dir + fname.split("/")[-1][:-4] + ".pdf")
            plt.cla()

    return reduce(lambda a, b: a + b, [p for _, p in peaks_of_order], []), energies, b_field, current
    # return peaks, energies, b_field, current


def main():
    # 14 - 26, 32, 31? 33?
    all_peaks = []
    all_es = []
    all_bs = []
    file_ids = []
    all_currents = []
    e_bs = []
    pvalue = 0.05
    save = argv[-1] == "save"
    for i in ["05", 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 32, 31, 33]:
        try:
            f = argv[1] + f"ZeemanX_0{i}.txt"
            peaks, e, b, current = process_file(f, pvalue=pvalue, preview=False, save=save)
            # if - b[0].nominal_value > 0.58:
            #     print(f)
            #     p, e, b = process_file(f, pvalue=pvalue, preview=True)
            #     break
            file_ids += [str(i) + " X"] * len(peaks)
            all_peaks += peaks
            all_es += e
            e_bs += [b for _ in range(len(e))]
            all_bs += [b for _ in range(len(peaks))]
            all_currents += [current for _ in range(len(peaks))]
            print(f"parsed {i} X")
        except Exception as e:
            print(e)
            raise e
        try:
            f = argv[1] + f"ZeemanY_0{i}.txt"
            peaks, e, b, current = process_file(f, pvalue=pvalue, preview=False, save=save)
            file_ids += [str(i) + " Y"] * len(peaks)
            all_peaks += peaks
            all_es += e
            all_bs += [b for _ in range(len(peaks))]
            e_bs += [b for _ in range(len(e))]
            all_currents += [current for _ in range(len(peaks))]
            print(f"parsed {i} Y")
        except Exception as e:
            print(e)
            raise e


    b_arr = np.array(list(map(lambda t: ~ (- t), e_bs)))
    b_err = np.array(list(map(lambda t: p.ve(t)[1], e_bs)))
    e_arr = np.array(list(map(lambda t: ~ np.abs(t), all_es)))
    e_err = np.array(list(map(lambda t: np.abs(p.ve(t)[1]), all_es)))

    params, (err, r_sq) = std.fit_func(lambda x, a: std.linear(x, a, 0), b_arr, e_arr)
    xrange = np.linspace(min(b_arr), max(b_arr), 1000)

    print("params: ", p.ev(params, err))

    plt.cla()
    plt.errorbar(b_arr, e_arr, xerr=b_err, yerr=e_err, **std.default.error_bar_def)
    plt.plot(xrange, std.linear(xrange, params[0], 0), label=f"$R^2 = {round(r_sq, 3)}$")
    std.default.plt_pretty("Magnetfeld / T", "Energieaufspaltung / eV")
    plt.legend()
    # plt.show()
    dir = "/".join(argv[0].split("/")[:-1]) + "/../figs/"
    if save:
        plt.savefig(dir + "magneton.pdf")
    else:
        plt.show()


    # print(len(all_peaks))
    # print(len(all_currents))
    dataset = {}
    dataset["Datei ID"] = file_ids
    dataset["Ordnung"] = list(map(lambda x: x.order, all_peaks))
    dataset["$\\mu / \\unit{\\meter}$"] = list(map(lambda x: x.position, all_peaks))
    dataset["$\\sigma$"] = list(map(lambda x: x.sigma, all_peaks))
    dataset["$A$"] = list(map(lambda x: x.height, all_peaks))
    dataset["$I / \\unit{\\ampere}$"] = all_currents
    dataset["$B / \\unit{\\tesla}$"] = all_bs

    if len(argv) > 2:
        std.print_tex_table(dataset, argv[2])


if __name__ == "__main__":
    main()
