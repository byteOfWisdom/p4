#!python3
import split


def main():
    peaks, _, b, _ = split.process_file("../data/zeeman/a203/ZeemanY_034.txt", preview=True)

    samples = [peaks[2], peaks[3]]

    hwhm = samples[0].fwhm / 2
    x0 = samples[0].position

    

    print(samples)

if __name__ == "__main__":
    main()
