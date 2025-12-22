import calibration
from sys import argv

import numpy as np
import propeller as p
import scipy
import std
from matplotlib import pyplot as plt

def gauss(x,a,mu,sigma,c):
    return a* np.exp( - (( ( (x-mu)**2) / (2*(sigma**2)))))+c

@np.vectorize
def square(x, a, b, c, d):
    if x < b:
        return d
    if x > c:
        return d
    return a-d

def main():

    position_err = 5e-4
    field_err = 10e-2

    data = np.transpose(np.loadtxt(argv[1],delimiter=",",skiprows=1))
    positions = p.ev(data[0], position_err)
    fields = p.ev(data[1], field_err)

    # middle_pos = p.ev(data[0][5:-5], position_err)
    # middle_fields = p.ev(data[1][5:-5], field_err)

    std.bullshit.ger()
    std.default.plt_pretty("Position / m", "Magnetfeld B / T")
    eb_param = std.default.error_bar_def

    #guesses = [0.5,0.195,0.204,0.005]

   #
   # params, cov = scipy.optimize.curve_fit(square, ~positions, ~fields, maxfev=9999,p0=guesses)

    #print(params)

    #xrange = np.linspace(min(~positions),max(~positions),1000)
    #plt.plot(xrange,square(xrange,*params))

    plt.errorbar(positions, fields, xerr=position_err, yerr=field_err, **eb_param)
    if len(argv) >= 3:
        plt.savefig(argv[2])
    else:
        plt.show()


if __name__ == "__main__":
    main()
