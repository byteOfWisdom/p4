#!python3
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
import std
import propeller as p


def load(file):
    data = np.transpose(np.loadtxt(file, delimiter="\t", skiprows=5))
    return data[2], data[1]


def main():
    list_file = std.readfile(argv[1])

    temps = [float(x.split(",")[0]) for x in list_file]
    u_max = [float(x.split(",")[1]) for x in list_file]
    u_2 = [float(x.split(",")[2]) for x in list_file]
    file = ["/".join(argv[1].split("/")[:-1]) + "/" + x.split(",")[3].strip() for x in list_file]

    use = [3, 15, 17, 23]

    for i in use:
        x, y = load(file[i])
        t = temps[i]

        plt.plot(x, y, marker="x", label=f"T = {t} ºC")
    plt.legend()
    std.default.plt_pretty("U_B / V", "U_I / V")
    plt.show()

    use = [1, 3, 5, 6, 8]
    for i in use:
        x, y = load(file[i])
        plt.plot(x, y, marker="x", label=f"$U_G = {u_2[i]} V$")
    plt.legend()
    std.default.plt_pretty("U_B / V", "U_I / V")
    plt.show()


if __name__ == "__main__":
    main()
