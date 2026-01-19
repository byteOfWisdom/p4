#!python3
import split
import std

def main():
    peaks, e, b, _ = split.process_file("../data/zeeman/a203/ZeemanY_034.txt", preview=False)
    # peaks, e, b, _ = split.process_file("../data/zeeman/a203/ZeemanY_012.txt", preview=True)

    lambda_zero = 643.8e-9
    samples = [peaks[2], peaks[3]]

    x0 = 0.5 * (samples[0].position + samples[1].position)
    fwhm = - lambda_zero * (1 - split.etalon_term(x0) / split.etalon_term(x0 - 0.5 * samples[0].fwhm))

    print(fwhm.format())

    # A = lambda_zero / delta_lambda
    h = std.unit.planck_const_eV
    c = std.unit.c
    µB = 5.7883818060e-5
    lambda_sigma = h * c / ((h * c / lambda_zero) + e[1])

    delta_lambda = lambda_zero - lambda_sigma

    F = delta_lambda / fwhm

    A = - µB * b * lambda_sigma / (h * c)

    print(F.format())
    print(A.format())

if __name__ == "__main__":
    main()
