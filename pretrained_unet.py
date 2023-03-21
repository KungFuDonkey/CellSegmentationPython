
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Concatenate, \
    BatchNormalization, Activation, RandomFlip, RandomRotation, Conv2DTranspose
from keras.models import Model


#Miss dat we ook nog de volgende al bestaande implementatie kunnen vergelijken:
#https://github.com/qubvel/segmentation_models


# Define the U-Net architecture with pretrained mobileNetV2 weights
# https://github.com/nikhilroxtomar/Unet-with-Pretrained-Encoder/blob/master/U-Net_with_Pretrained_MobileNetV2_as_Encoder.ipynb
def pretrained_unet(input_shape=(480, 640, 3), f=None, n_classes=1):
    if f is None:
        f = [3, 96, 144, 192] #Number of filters per layer in the decoder (reverse order)

    inputs = tf.keras.Input(shape=input_shape, batch_size=64)


    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Load the pre-trained MobileNetV2 model as the encoder
    encoder = tf.keras.applications.MobileNetV2(input_tensor=x, include_top=False, weights='imagenet')
    # encoder.summary()
    # input_1 (InputLayer)           [(None, 480, 640 ,3)]
    # block_1_expand_relu (ReLU)     (None, 240, 320, 96)
    # block_3_expand_relu (ReLU)     (None, 120, 160, 144)
    # block_6_expand_relu (ReLU)     (None,  60, 80, 192)
    encoder.trainable = False

    skip_connection_names = ["input_1", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]

    x = encoder.get_layer("block_13_expand_relu").output

    for i in range(1, len(skip_connection_names) + 1):
        #Can unify this code with the upsampling block from the manual unet later
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = Conv2DTranspose(
            f[-i],  # number of filters
            (3, 3),  # Kernel size
            strides=(2, 2),
            padding="same")(x)
        merge = Concatenate()([x, x_skip])

        x = Conv2D(f[-i], (3, 3), padding="same", kernel_initializer='he_normal')(merge)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f[-i], (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    # #Extra convolution at the end as done in the manual model
    # x = Conv2D(f[0], (3, 3), padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)

    output = Conv2D(n_classes, (1, 1), padding="same", kernel_initializer='he_normal')(x)
    model = Model(inputs, output) # need to compile with from_logits=True
    return model





