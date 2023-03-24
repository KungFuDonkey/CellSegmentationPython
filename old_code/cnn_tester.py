from pretrained_unet import pretrained_unet_model
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
groundtruths = [np.sum(gt, axis=-1) // 3 for gt in groundtruths]

resized_images = np.float32(resize_images(cv_images, shape=(224, 224, 3)))
resized_groundtruths = np.float32(resize_images(groundtruths, shape=(224, 224, 1)))

X_train = resized_images[:60]
Y_train = resized_groundtruths[:60]

X_test = resized_images[60:]
Y_test = resized_groundtruths[60:]

# IMAGES ARE ACTUALLY GREYSCALE EVEN THOUGH THERE ARE 3 CHANNELS

load_model = False
batch_size = 20
epochs = 10
plot_history = False
man_model_path = "saved_manual_model"
pre_model_path = "saved_pretrained_model"
pre_model_load_path = "saved_pretrained_model_b20_e10"
man_callback = tf.keras.callbacks.ModelCheckpoint(filepath=man_model_path, verbose=1)
pre_callback = tf.keras.callbacks.ModelCheckpoint(filepath=pre_model_path, verbose=1)


# pre_unet = pretrained_unet_model()
# pre_unet.compile(optimizer='adam',
#                  loss=tf.keras.losses.binary_crossentropy,
#                  metrics=['accuracy'])
# pre_unet.summary()
# if load_model:
#     pre_unet.load_weights(pre_model_load_path)
#     history = pre_unet.history
# else:
#     history = pre_unet.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[pre_callback])
#
# pred_nonzero = np.nonzero((pre_unet(X_train[:1]).numpy() > 0.5)[0])
# true_nonzero = np.nonzero(Y_train[0])
# print("done with pretrained model")

# resized_images = resize_images(cv_images, shape=(480, 640, 3))
# resized_groundtruths = resize_images(groundtruths, shape=(480, 640, 1))
#
# X_train = resized_images[:60]
# Y_train = resized_groundtruths[:60]
#
# X_test = resized_images[60:]
# Y_test = resized_groundtruths[60:]

# man_unet = unet_model()
# man_unet.compile(optimizer='adam',
#                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                  metrics=['accuracy'])
# #man_unet.load_weights(man_model_path)
# #history = man_unet.history
# history = man_unet.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
#                        validation_data=(X_test, Y_test), callbacks=[man_callback]
#                        )


# man_unet.predict(X_train[:5])

if plot_history:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
