import os
import random
import cv2

class Augmentator:
    def __init__(self) -> None:
        self.base_dir = None
        self.opened_files = []
        self.opened_labels = []
        self.statistic_example_image_path = None

    def load_images(self, path, percent):
        self.base_dir = path
        self.opened_files = []
        self.opened_labels = []

        for subdir, dirs, files in os.walk(path):
            label = subdir[len(path):]
            
            files = [file for file in files if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]

            if files:
                random.shuffle(files)

                chosen_files = files[:int(percent / 100 * len(files))]

                self.opened_files.extend([
                    cv2.cvtColor(cv2.imread(os.path.join(subdir, file)), cv2.COLOR_BGR2GRAY) \
                        for file in chosen_files])
                self.opened_labels.extend([os.path.join(label, file) for file in chosen_files])