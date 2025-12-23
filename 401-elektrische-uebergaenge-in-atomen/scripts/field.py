from split import calib_curve
import numpy as np
import std
from sys import argv
import propeller as p


def main():
    current = float(argv[1])
    field, field_err = p.ve(calib_curve(current))
    print(field,field_err)


if __name__ == "__main__":
    main()
