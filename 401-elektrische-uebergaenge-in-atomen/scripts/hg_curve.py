import numpy as np
from sys import argv
import std
from matplotlib import pyplot as plt


def main():
        pressure_nolog = np.linspace(0,100)
        offset = 10.55
        temperatures_k = np.linspace(200,500,500)
        #temperatures_c = temperatures_k - 173.15
        pressure_log = lambda t: offset - (3333 / t) - (0.85 * np.log(t))
        #pressure_nolog = np.exp(pressure_log(temperatures_k))
        pressure_pa = lambda t: np.exp(pressure_log(t)) * 133
        start_temp = 160 + 173.15
        end_temp = 180 + 173.15
        test_temps = np.linspace(start_temp,end_temp,500)
        std.default.plt_pretty ("Temperatur T / K", "Druck p / Pa")
        plt.plot(temperatures_k, pressure_pa(temperatures_k),label="Hg-Dampfdruckkurve",linestyle="--")
        plt.plot(test_temps,pressure_pa(test_temps),label = "Hg-Dampfdruckkurve im gemessenen Intervall")
        plt.legend(loc="best")
        plt.yscale("log")
        if len(argv) >= 2:
            plt.savefig(argv[1])
        else:
            plt.show()


if __name__ == "__main__":
    main()
