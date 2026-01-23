#!python3

from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
from sys import argv
import std


def is_pink(rgb):
    # return np.all(rgb == [255, 0, 255, 255])
    return (rgb[0] > 200) & (rgb[1] < 95) & (rgb[2] > 200) & (rgb[3] == 255)


def main():
    im_data = np.array(image.imread(argv[1]))

    is_point = np.apply_along_axis(is_pink, 2, im_data)
    y, x = np.where(is_point)
    # print(points)

    plt.imshow(im_data)
    plt.scatter(x, y, marker="x")
    plt.show()
    

if __name__ == "__main__":
    main()
