# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt


def find_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def apply_watershed(cv_image):
    gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # invert = cv.bitwise_not(thresh)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)


    # Finding sure foreground area
    dist_transform = np.uint16(cv.distanceTransform(opening, cv.DIST_L2, 5))
    dist_transform = cv.morphologyEx(dist_transform, cv.MORPH_OPEN, kernel, iterations=2)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # ret, sure_fg = cv.threshold(dist_transform, 0 * dist_transform.max(), 255, cv.THRESH_BINARY)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(cv_image, markers)
    cv_image[markers == -1] = [255, 0, 0]
    cv.imshow('thresh', thresh)
    cv.imshow('opening', opening)
    cv.imshow('sure_bg', sure_bg)
    cv.imshow('sure_fg', sure_fg)


def display_image(cv_image):
    cv.imshow('', cv_image)
    cv.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_images = find_images('inputImages')
    for image_path in input_images:
        cv_image = cv.imread(image_path)
        apply_watershed(cv_image)
        display_image(cv_image)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
