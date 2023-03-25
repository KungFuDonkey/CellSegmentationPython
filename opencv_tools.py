import cv2 as cv
import os
import numpy as np
import random as rnd

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
    return [(np.sum(image, axis=-1) > 0).astype(int) for image in input_images]


def augment_images(raw_images, groundtruth_images, max_shift=0.7, max_angle=90, max_zoom_in=3, max_zoom_out=0.5):
    augmented_raw = []
    augmented_ground = []

    # Augment every pair of (raw, groundtruth) with the same random transformation
    for raw, ground in zip(raw_images, groundtruth_images):
        rnd_shift = rnd.uniform(-max_shift, max_shift)
        rnd_angle = int(rnd.uniform(-max_angle, max_angle))
        rnd_zoom_in = rnd.uniform(1, max_zoom_in)
        rnd_zoom_out = rnd.uniform(max_zoom_out, 1)

        augmented_raw.append(augment_image(raw, rnd_shift, rnd_angle, rnd_zoom_in, rnd_zoom_out))
        augmented_ground.append(augment_image(ground, rnd_shift, rnd_angle, rnd_zoom_in, rnd_zoom_out))
    return np.array(augmented_raw).flatten(), np.array(augmented_ground).flatten()


# Turns a single image into 8 images (including the original)
def augment_image(img, shift, angle, zoom_in, zoom_out):
    return [
        img,
        horizontal_shift(img, shift),
        vertical_shift(img, shift),
        horizontal_flip(img),
        vertical_flip(img),
        rotate(img, angle),
        zoom(img, zoom_in),
        zoom(img, zoom_out)
    ]


def horizontal_shift(img, shift):
    h, w = img.shape[:2]
    trans_mat = np.float32([[1, 0, w * shift], [0, 1, 0]])
    return cv.warpAffine(img, trans_mat, (w, h))


def vertical_shift(img, shift):
    h, w = img.shape[:2]
    trans_mat = np.float32([[1, 0, 0], [0, 1, h * shift]])
    return cv.warpAffine(img, trans_mat, (w, h))


def horizontal_flip(img):
    return cv.flip(img, 1)


def vertical_flip(img):
    return cv.flip(img, 0)


def rotate(img, angle):
    h, w = img.shape[:2]
    rot_mat = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv.warpAffine(img, rot_mat, (w, h))
    return img


def zoom(img, zoom):
    h, w = img.shape[:2]
    zoom_mat = cv.getRotationMatrix2D((w / 2, h / 2), 0, zoom)
    return cv.warpAffine(img, zoom_mat, img.shape[1::-1], cv.INTER_LINEAR)
