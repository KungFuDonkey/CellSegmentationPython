import cv2 as cv
import os
import numpy as np


IMAGE_EXPORT_TYPE = '.bmp'


# finds images from the folder
def find_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


# finds images from the folder in opencv format
def find_opencv_images(folder):
    input_images = find_images(folder)
    return list(map(cv.imread, input_images))


# displays an image and waits until you exit the window
def display_image(cv_image):
    cv.imshow('', np.float32(cv_image))
    cv.waitKey(0)


# exports an image to the outputImages folder
def export_image(cv_image, method_name, image_name):
    output_folder_path = 'outputImages'
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    method_folder_path = os.path.join(output_folder_path, method_name)
    if not os.path.isdir(method_folder_path):
        os.mkdir(method_folder_path)

    image_path = os.path.join(method_folder_path, image_name + IMAGE_EXPORT_TYPE)

    cv.imwrite(image_path, cv_image)


def make_binary_images(input_images):
    return [(np.sum(image, axis=-1)>0).astype(int) for image in input_images]