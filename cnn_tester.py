from pretrained_unet import pretrained_unet
from manual_unet import unet_model
import tensorflow as tf
import opencv_tools
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from skimage.transform import resize

def resize_images(images, shape):
    return np.array([resize(im, shape, order=0, preserve_range=True, anti_aliasing=False) for im in images])

cv_images = opencv_tools.find_opencv_images('dataset//rawimages')
groundtruths = opencv_tools.find_opencv_images('dataset//groundtruth')
groundtruths = [np.sum(gt, axis=-1)//3 for gt in groundtruths]

resized_images = resize_images(cv_images, shape =(480, 640,3))
resized_groundtruths = resize_images(groundtruths, shape =(480, 640,1))

X_train = resized_images[:60]
Y_train = resized_groundtruths[:60]

X_test = resized_images[60:]
Y_test = resized_groundtruths[60:]

unet = unet_model()
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
unet.summary()
history = unet.fit(X_train, Y_train, epochs=2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()