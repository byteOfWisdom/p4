#!python3

from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
from sys import argv
import std


def is_pink(rgb):
    # return np.all(rgb == [255, 0, 255, 255])
    return (rgb[0] > 200) & (rgb[1] < 95) & (rgb[2] > 200) & (rgb[3] == 255)


def find_laue_maxima(im_data):
    is_point = np.apply_along_axis(is_pink, 2, im_data)
    y, x = np.where(is_point)

    #find middle:
    points = np.transpose(np.array([x, y]))
    avg = np.average(points, 0)
    dists = np.sum((points - avg) ** 2, 1)
    middle = np.where(dists - np.min(dists) < 1000)[0]
    middle_coord = np.average(points[middle], 0)
    points = np.delete(points, middle, 0)

    pixel_spacing = 1 # todo: measure the fucking film
    points = points - middle_coord
    points *= pixel_spacing

    return points


def dedup_double_markings(points):
    i = 0
    while i < len(points):
        p = points[i]
        dists = np.sqrt(np.sum((points - p) ** 2, 1))
        print(np.sort(dists))
        points = np.delete(points, (dists != 0) & (dists <= 1.5), 0)
        i += 1

    return points


def main():
    im_data = np.array(image.imread(argv[1]))

    points = find_laue_maxima(im_data)
    points = dedup_double_markings(points)

    # plt.imshow(im_data)
    std.default.plt_pretty("x", "y")
    plt.gca().set_aspect('equal')
    plt.scatter(np.transpose(points)[0], np.transpose(points)[1], marker="x")
    plt.show()

    print(points)
    

if __name__ == "__main__":
    main()
