from collections import deque

import numpy as np

from images import BinaryImage, HalftoneImage, ColourImage


def count_distances(arr: np.ndarray) -> np.ndarray:
    distances = np.where(arr, 0, 1_000_000)
    visited = arr.astype(int) * 2

    d = deque(np.argwhere(arr))

    for y, x in np.argwhere(arr):
        visited[y, x] = max(visited[y, x], 1)

    while d:
        y, x = d.popleft()
        visited[y, x] = 2

        min_dist = distances[y, x]

        if y > 0:
            min_dist = min(min_dist, distances[y - 1, x] + 1)
            if visited[y - 1, x] == 0:
                d.append([y - 1, x])
                visited[y - 1, x] = 1
        if y + 1 < arr.shape[0]:
            min_dist = min(min_dist, distances[y + 1, x] + 1)
            if visited[y + 1, x] == 0:
                d.append([y + 1, x])
                visited[y + 1, x] = 1
        if x > 0:
            min_dist = min(min_dist, distances[y, x - 1] + 1)
            if visited[y, x - 1] == 0:
                d.append([y, x - 1])
                visited[y, x - 1] = 1
        if x + 1 < arr.shape[1]:
            min_dist = min(min_dist, distances[y, x + 1] + 1)
            if visited[y, x + 1] == 0:
                d.append([y, x + 1])
                visited[y, x + 1] = 1

        distances[y, x] = min_dist

    return distances


class ImageConverter:
    @classmethod
    def binary_to_binary(cls, img: BinaryImage) -> BinaryImage:
        return BinaryImage(img.pixels)

    @classmethod
    def binary_to_halftone(cls, img: BinaryImage) -> HalftoneImage:
        distances_to_white = count_distances(img.pixels)
        distances_to_black = count_distances(1 - img.pixels)

        norm_dist_to_white = 1 / (distances_to_white + 1)
        norm_dist_to_black = distances_to_black / np.max(distances_to_black)

        return HalftoneImage(
            (norm_dist_to_white + norm_dist_to_black) * 255 / 2)

    @classmethod
    def binary_to_colour(cls, img: BinaryImage,
                         palette: np.ndarray) -> ColourImage:
        return cls.halftone_to_colour(cls.binary_to_halftone(img), palette)

    @classmethod
    def halftone_to_binary(cls, img: HalftoneImage,
                           threshold: int = 127) -> BinaryImage:
        return HalftoneImage(img.pixels > threshold)

    @classmethod
    def halftone_to_halftone(cls, img: HalftoneImage,
                             mean: float, var: float) -> HalftoneImage:
        old_mean = img.pixels.mean()
        old_var = img.pixels.var()
        print(old_mean, old_var)
        new_img = mean + (img.pixels - old_mean) * var / old_var

        return HalftoneImage(np.clip(new_img, 0, 255).astype(int))

    @classmethod
    def halftone_to_colour(cls, img: HalftoneImage,
                           palette: np.ndarray) -> ColourImage:
        assert isinstance(palette, np.ndarray)
        assert palette.shape == (256, 3)

        return ColourImage(palette[img.pixels])

    @classmethod
    def colour_to_binary(cls, img: ColourImage,
                         threshold: int = 127) -> BinaryImage:
        return cls.halftone_to_binary(cls.colour_to_halftone(img), threshold)

    @classmethod
    def colour_to_halftone(cls, img: ColourImage) -> HalftoneImage:
        return HalftoneImage(img.pixels.mean(axis=2))

    @classmethod
    def colour_to_colour(cls, img: ColourImage,
                         mean: float, var: float) -> ColourImage:

        old_mean = img.pixels.mean(axis=(0, 1))
        old_var = img.pixels.var(axis=(0, 1))
        new_img = mean + (img.pixels - old_mean) * var / old_var

        return ColourImage(np.clip(new_img, 0, 255).astype(int))
