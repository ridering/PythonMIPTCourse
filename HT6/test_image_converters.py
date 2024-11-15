import os

import cv2
import numpy as np

from images import BinaryImage, HalftoneImage, ColourImage
from image_converter import ImageConverter

INPUT_DIR = './input_data/'
OUTPUT_DIR = './output_data/'

input_image = ColourImage(
    cv2.imread(INPUT_DIR + 'image.png', cv2.COLOR_BGR2RGB))


binary_image = ImageConverter.colour_to_binary(input_image, 127)
cv2.imwrite(OUTPUT_DIR + 'binary_from_colour.png', binary_image.pixels * 255)

halftone_image = ImageConverter.colour_to_halftone(input_image)
cv2.imwrite(OUTPUT_DIR + 'halftone_from_colour.png', halftone_image.pixels)

colour_from_colour = ImageConverter.colour_to_colour(input_image, 90, 3000)
cv2.imwrite(OUTPUT_DIR + 'colour_from_colour.png', colour_from_colour.pixels)


binary_from_halftone = ImageConverter.halftone_to_binary(halftone_image, 127)
cv2.imwrite(OUTPUT_DIR + 'binary_from_halftone.png',
            binary_from_halftone.pixels * 255)

halftone_from_halftone = ImageConverter.halftone_to_halftone(
    halftone_image, 40, 2000)
cv2.imwrite(OUTPUT_DIR + 'halftone_from_halftone.png',
            halftone_from_halftone.pixels)

palette = np.random.randint(0, 256, (256, 3))
color_from_halftone = ImageConverter.halftone_to_colour(
    halftone_image, palette)
cv2.imwrite(OUTPUT_DIR + 'colour_from_halftone.png',
            color_from_halftone.pixels)


binary_from_binary = ImageConverter.binary_to_binary(binary_image)
cv2.imwrite(OUTPUT_DIR + 'binary_from_binary.png',
            binary_from_binary.pixels * 255)

halftone_from_binary = ImageConverter.binary_to_halftone(binary_image)
cv2.imwrite(OUTPUT_DIR + 'halftone_from_binary.png',
            halftone_from_binary.pixels)

colour_from_binary = ImageConverter.binary_to_colour(binary_image, palette)
cv2.imwrite(OUTPUT_DIR + 'colour_from_binary.png', colour_from_binary.pixels)


# print(colour_from_colour.pixels)
