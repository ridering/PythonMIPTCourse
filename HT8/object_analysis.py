"""
Шаблонный метод (Template method)
"""
import cv2
import numpy as np
import random as rng
import pandas as pd


def calculate_hu_moments(func):
    def wrapper(*args, **kwargs):
        canny_output = func(*args, **kwargs)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        mu = [None] * len(contours)
        for idx in range(len(contours)):
            mu[idx] = cv2.moments(contours[idx])

        mc = [None] * len(contours)
        for idx in range(len(contours)):
            mc[idx] = (mu[idx]['m10'] / (mu[idx]['m00'] + 1e-5),
                       mu[idx]['m01'] / (mu[idx]['m00'] + 1e-5))

        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3),
                           dtype=np.uint8)

        for idx in range(len(contours)):
            color = (
                rng.randint(
                    0, 256), rng.randint(
                    0, 256), rng.randint(
                    0, 256))
            cv2.drawContours(drawing, contours, idx, color, 2)
            cv2.circle(drawing, (int(mc[idx][0]), int(mc[idx][1])), 4, color,
                       -1)
        cv2.imwrite('output_data/colored.png', drawing)
        f = open('output_data/output.txt', 'w')
        for idx in range(len(contours)):
            print(
                ' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - '
                'Length: %.2f' %
                (idx, mu[idx]['m00'], cv2.contourArea(
                    contours[idx]), cv2.arcLength(
                    contours[idx], True)), file=f)
        f.close()
        return canny_output, mu, mc
    return wrapper


class ObjectAnalysis:
    def gauss_filter(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    @calculate_hu_moments
    def canny_edge(self, img):
        return cv2.Canny(img, 100, 200)

    def segmentation(self, img):
        img[np.all(img == 255, axis=2)] = 0

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
        sharp = np.float32(img)
        imgResult = sharp - imgLaplacian

        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype(np.uint8)
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)

        bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw.astype(np.uint8), 40, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

        kernel1 = np.ones((3, 3), dtype=np.uint8)
        dist = cv2.dilate(dist, kernel1)

        dist_8u = dist.astype(np.uint8)
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        markers = np.zeros(dist.shape, dtype=np.int32)
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i + 1), -1)

        cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
        markers_8u = (markers * 10).astype('uint8')

        cv2.watershed(imgResult, markers)
        mark = markers.astype('uint8')
        mark = cv2.bitwise_not(mark)

        colors = []
        for contour in contours:
            colors.append((rng.randint(0, 256), rng.randint(0, 256),
                           rng.randint(0, 256)))
        dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i, j]
                if index > 0 and index <= len(contours):
                    dst[i, j, :] = colors[index - 1]

        return dst

    def template_method(self, image):

        image = self.noise_filtering(image)
        data = self.segmentation(image)
        data = self.object_parameters(data)

        return data

    def noise_filtering(self, image):
        b, g, r = cv2.split(image)
        blur_b = cv2.GaussianBlur(b, self.get_kernel_size(), 0)
        blur_g = cv2.GaussianBlur(g, self.get_kernel_size(), 0)
        blur_r = cv2.GaussianBlur(r, self.get_kernel_size(), 0)
        blur_img = cv2.merge([blur_b, blur_g, blur_r])
        return blur_img

    def object_parameters(self, data):
        (image, data) = data
        (numLabels, labels, stats, centroids) = data
        x = []
        y = []
        w = []
        h = []
        area = []
        for i in range(1, numLabels):
            # extract the connected component statistics for the current
            # label
            x.append(stats[i, cv2.CC_STAT_LEFT])
            y.append(stats[i, cv2.CC_STAT_TOP])
            w.append(stats[i, cv2.CC_STAT_WIDTH])
            h.append(stats[i, cv2.CC_STAT_HEIGHT])
            area.append(stats[i, cv2.CC_STAT_AREA])

        return (x, y, w, h, area)


class BinaryImage(ObjectAnalysis):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        median = cv2.medianBlur(image, 5)
        return median


class MonochromeImage(BinaryImage):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        return None


class ColorImage(MonochromeImage):
    def __init__(self):
        pass


if __name__ == '__main__':
    color_conv = ColorImage()
    img = cv2.imread("input_data/input_image.png")
    contours, mu, mc = color_conv.canny_edge(img)

    seg = color_conv.segmentation(img)

    cv2.imwrite("output_data/contours.png", contours)
    cv2.imwrite("output_data/segmented.png", seg)

    
    with open("output_data/mu.txt", 'w') as file:
        for elem in mu:
            print(elem, file=file)
    with open("output_data/mc.txt", 'w') as file:
        for elem in mc:
            print(elem, file=file)

