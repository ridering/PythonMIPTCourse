import cv2
import numpy as np

class AbstractFactoryImageReader():
    def __init__(self, im_reader):
        self.__im_reader = im_reader

    def read_image(self, file_path):
        self.__im_reader.read_image(file_path)

class BinaryImageReader:
    def read_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        binary = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
        self.__image = binary.astype(np.uint8)

class HalftoneImageReader:
    def read_image(self, file_path):
        self.__image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

class ColourImageReader:
    def read_image(self, file_path):
        self.__image = cv2.imread(file_path)

def get_image_reader(ident):
    if ident == 0:
        return BinaryImageReader()
    elif ident == 1:
        return HalftoneImageReader()
    elif ident == 2:
        return ColourImageReader()

if __name__ == "__main__":
    readers = [
        AbstractFactoryImageReader(BinaryImageReader()),
        AbstractFactoryImageReader(HalftoneImageReader()),
        AbstractFactoryImageReader(ColourImageReader()),
    ]

    for reader in readers:
        print(reader)