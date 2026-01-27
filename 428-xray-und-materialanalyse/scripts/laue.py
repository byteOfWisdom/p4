#!python3

from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
from sys import argv
import std
import propeller as p


def is_pink(rgb):
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

    points = points - middle_coord

    return points


def dedup_double_markings(points):
    i = 0
    while i < len(points):
        p = points[i]
        dists = np.sqrt(np.sum((points - p) ** 2, 1))
        points = np.delete(points, (dists != 0) & (dists <= 1.5), 0)
        i += 1

    return points


def possible_lattice_vectors(limit):
    if limit % 2:
        limit -= 1
    evens = np.append(np.arange(0, limit + 1, 2), -1 * np.arange(0, limit + 1, 2))
    # positive_evens = np.arange(0, limit + 1, 2)
    positive_evens = np.array([0, 2])
    odds = np.append(np.arange(1, limit + 2, 2), -1 * np.arange(1, limit + 2, 2))
    # positive_odds = np.arange(1, limit + 2, 2)
    positive_odds = np.array([1, 3])
    h, j, k = np.meshgrid(evens, evens, positive_evens)
    even_tuples = [np.ravel(h), np.ravel(j), np.ravel(k)]
    h, j, k = np.meshgrid(odds, odds, positive_odds)
    odd_tuples = [np.ravel(h), np.ravel(j), np.ravel(k)]
    vecs = np.transpose(np.append(even_tuples, odd_tuples, 1))
    not_zero = [v[0] != 0 and v[1] != 0 and v[2] != 0 for v in vecs]
    return vecs[not_zero]


def assign_miller_indices(points):
    dist = 15e-3 # 15mm distance
    zq = np.sqrt(np.sum(points ** 2, 1) + dist) - dist

    points_3d = np.array([[points[i][0], points[i][1], zq[i]] for i in range(len(points))])

    candidates = possible_lattice_vectors(6)
    lens = np.sum(candidates ** 2, -1) ** -0.5
    norm_candidates = np.array([lens[i] * candidates[i] for i in range(len(candidates))])
    lens_p = np.sum(points_3d ** 2, -1) ** -0.5
    norm_points = np.array([lens_p[i] * candidates[i] for i in range(len(points_3d))])

    res = []

    for p in norm_points:
        deviation = np.sum(np.cross(p, norm_candidates) ** 2, -1) * np.sum(np.abs(candidates), -1)
        id = np.where(np.isclose(deviation, np.min(deviation)))[0][0]
        grating_vec = candidates[id]
        res.append(grating_vec)

    return np.array(res)


def scale_points(points, im_data):
    # film dimensions are: 
    film_height = 76e-3

    pixel_spacing = film_height / np.max(np.shape(im_data))
    points *= pixel_spacing
    return points


def angle(miller_vec):
    h, k, l = np.transpose(miller_vec)
    return np.arctan(l / np.sqrt(h * h + k * k))


def distance(miller_vec):
    a0 = 564e-12
    miller_len = np.sqrt(np.sum(miller_vec ** 2, -1))
    d = a0 / miller_len
    return d


def wavelength(theta, d):
    return 2 * np.sin(theta) * d

def main():
    im_data = np.array(image.imread(argv[1]))

    points = find_laue_maxima(im_data)
    points = dedup_double_markings(points)
    points = scale_points(points, im_data)

    miller_indices = assign_miller_indices(points)
    distances = distance(miller_indices)
    angles = angle(miller_indices)
    wavelengths = wavelength(angles, distances)
    print("d are:")
    print(distances)

    print("angles are:")
    print(np.rad2deg(angles))
    print(angles)

    print("wavelengths are:")
    print(wavelengths * 1e9)

    print(miller_indices)

    tmi = np.transpose(miller_indices)
    h, k, l = tmi[0], tmi[1], tmi[2]

    indices = np.arange(len(h))

    point_and_index_table = {
        "Punkt Nr.": indices,
        "x / mm": p.ev(1e3 * np.transpose(points)[0], 2),
        "y / mm": p.ev(1e3 * np.transpose(points)[1], 2),
        "h": h,
        "k": k,
        "l": l
    }

    physical_things_table = {
        "Punkt Nr.": indices,
        "d / pm": distances * 1e12,
        "$\\theta$ / rad": angles,
        "$\\lambda$ / pm": wavelengths * 1e12
    }

    std.util.print_tex_table(point_and_index_table, "../latex/miller_indices.table")
    std.util.print_tex_table(physical_things_table, "../latex/miller_measurements.table")

    std.default.plt_pretty("x / mm", "y / mm")
    plt.gca().set_aspect('equal')
    plt.scatter(1e3 * np.transpose(points)[0], 1e3 * np.transpose(points)[1], marker="x")

    for i in range(len(miller_indices)):
        plt.annotate(f"({miller_indices[i][0]}, {miller_indices[i][1]}, {miller_indices[i][2]})", (1e3 * points[i][0], 1e3 * points[i][1]))
    plt.show()

if __name__ == "__main__":
    main()
