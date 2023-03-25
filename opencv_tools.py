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


def augment_images(raw_images, groundtruth_images):
    for img in groundtruth_images:
        [raw, _, _, _, _, rot, zoom] = augment_image(img)
        cv.imshow('raw', np.float32(raw))
        cv.imshow('rot', np.float32(rot))
        cv.imshow('zoom', np.float32(zoom))
        cv.waitKey(0)
    return map(augment_image, groundtruth_images)


def augment_image(raw_image):
    return [
        raw_image,
        horizontal_shift(raw_image, 0.7),
        vertical_shift(raw_image, 0.7),
        horizontal_flip(raw_image),
        vertical_flip(raw_image),
        rotate(raw_image, 30),
        zoom(raw_image, 0.5)
    ]


def horizontal_shift(img, max_shift=0.0):
    if max_shift > 1 or max_shift < 0:
        return img

    h, w = img.shape[:2]
    to_shift = w * rnd.uniform(-max_shift, max_shift)
    trans_mat = np.float32([[1,0,to_shift],[0,1,0]])
    return cv.warpAffine(img, trans_mat, (w,h))


def vertical_shift(img, max_shift=0.0):
    if max_shift > 1 or max_shift < 0:
        return img

    h, w = img.shape[:2]
    to_shift = h * rnd.uniform(-max_shift, max_shift)
    trans_mat = np.float32([[1, 0, 0], [0, 1, to_shift]])
    return cv.warpAffine(img, trans_mat, (w, h))


def horizontal_flip(img):
    return cv.flip(img, 1)


def vertical_flip(img):
    return cv.flip(img, 0)


def rotate(img, angle):
    angle = int(rnd.uniform(-angle, angle))
    h, w = img.shape[:2]
    rot_mat = cv.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle,1)
    img = cv.warpAffine(img, rot_mat, (w, h))
    return img


def zoom(img, zoom):
    h, w = img.shape[:2]
    zoom_mat = cv.getRotationMatrix2D((w/2, h/2), 0, zoom)
    return cv.warpAffine(img, zoom_mat, img.shape[1::-1], cv.INTER_LINEAR)
