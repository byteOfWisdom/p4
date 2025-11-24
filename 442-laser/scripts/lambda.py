#!python3
from sys import argv

import numpy as np
import propeller as p
import scipy
import std
from matplotlib import pyplot as plt

# std.bullshit.this_is_fucking_stupid_no_one_actually_gives_a_fuck()


def main():
    data = np.transpose(np.loadtxt(argv[1], delimiter=",", skiprows=1))
    # measure_num = data[0][:3]
    order = data[1]
    order -= 1
    translation = p.ev(data[2], 5e-3)
    translation = translation[0] - translation

    distance = float(argv[2])
    grating_const = float(argv[3])
    angle_term = np.sin(np.arctan(translation / distance))
    angle_term_v, angle_term_e = p.ve(grating_const * angle_term)
    fmt = std.default.error_bar_def
    fmt["fmt"] = "x"
    plt.errorbar(order, angle_term_v, yerr=angle_term_e, **fmt)
    std.default.plt_pretty("Ordnung", r"$g \cdot \sin(\arctan(x / d))$ / $mm^{-1}$")

    res, meta = std.fit_func(
        lambda x, a: a * x, order, angle_term_v, y_errors=angle_term_e
    )

    x = np.linspace(0, 2, 100)

    r_sq = std.goodness_of_fit(res[0] * order, angle_term_v)
    plt.plot(x, x * res[0], label=f"$R^2 = {round(r_sq, 3)}$")

    print(f"{res[0]} +- {meta.sd_beta[0]}")

    plt.legend()
    if len(argv) > 4:
        plt.savefig(argv[4])
    else:
        plt.show()


if __name__ == "__main__":
    main()
