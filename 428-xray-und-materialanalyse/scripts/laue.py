#!python3

from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
from sys import argv
import std


def binary_constrast(data, cutoff=None):
    if not std.some(cutoff):
        cutoff = 128
    high = data > cutoff
    data[high] = 255
    data[~high] = 0
    return data


def to_bw(data):
    return np.average(data, axis=2)


def crop_sqaure(data, offset=0):
    dim = np.shape(data)
    width = min(dim[0], dim[1])
    height = max(dim[0], dim[1])
    middle = int(0.5 * height) - offset
    hw = int(0.5 * width)
    return data[...][middle - hw:middle + hw]



def main():
    im_data = np.array(image.imread(argv[1]))
    im_data = to_bw(im_data)
    im_data = crop_sqaure(im_data)
    # plt.imshow(im_data)
    # plt.show()

    high_contrast = binary_constrast(im_data, 200)
    plt.imshow(high_contrast)
    plt.show()

    

if __name__ == "__main__":
    main()
