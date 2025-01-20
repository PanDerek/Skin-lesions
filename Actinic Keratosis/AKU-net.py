from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, BatchNormalization, Activation, Reshape, concatenate
from tensorflow.keras.models import Model
import numpy as np

def Unet(input_size=(256,256,3), dropout_rate=0.6):
    # Load the VGG-16 model and its pre-trained weights
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_size)
    vgg16_layers = dict([(layer.name, layer) for layer in vgg16.layers])

    N = input_size[0]
    inputs = Input(input_size)

    # Use the initial layers of VGG-16 as the first two convolutional layers of UNet
    conv1 = vgg16_layers['block1_conv1'](inputs)
    conv1 = vgg16_layers['block1_conv2'](conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Use the remaining layers of VGG-16 as the next convolutional layers of UNet
    conv2 = vgg16_layers['block2_conv1'](pool1)
    conv2 = vgg16_layers['block2_conv2'](conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = vgg16_layers['block3_conv1'](pool2)
    conv3 = vgg16_layers['block3_conv2'](conv3)
    conv3 = vgg16_layers['block3_conv3'](conv3)
    drop3 = Dropout(dropout_rate)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = vgg16_layers['block4_conv1'](pool3)
    conv4 = vgg16_layers['block4_conv2'](conv4)
    conv4 = vgg16_layers['block4_conv3'](conv4)
    drop4 = Dropout(dropout_rate)(conv4)


    conv5 = vgg16_layers['block5_conv1'](drop4)
    conv5 = vgg16_layers['block5_conv2'](conv5)
    conv5 = vgg16_layers['block5_conv3'](conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop5)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    merge6  = concatenate([drop3, up6], axis = 3)

    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    merge7  = concatenate([conv2, up7], axis = 3)

    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)

    merge8  = concatenate([conv1, up8], axis = 3)

    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(inputs = inputs, outputs = conv9)

    return model
