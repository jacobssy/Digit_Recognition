from utils import conv,pool,dropout
from keras.layers import Dense,Input, Flatten,concatenate
from keras.models import Model,Sequential
import tensorflow as tf
import numpy as np
from keras.layers import Activation, BatchNormalization,Dropout,add
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.regularizers import l2

NUM_CLASSES =10

def cnn_model0(IMG_SIZE):
    data_input = Input(shape=(IMG_SIZE, IMG_SIZE,3))
    layer1 = conv(data_input, 32, (5, 5), padding='same', strides=(1, 1), name='layer1')
    layer2 = pool(layer1, pool_size=(3, 3), strides=(2, 2), padding='same', name='layer2', pool_type='max')
    layer3 = conv(layer2, 64, (3, 3), padding='same', strides=(1, 1), name='layer3')
    layer4 = conv(layer3, 96, (3, 3), padding='same', strides=(1, 1), name='layer4')
    layer5 = pool(layer4,pool_size=(3, 3), strides=(2, 2), padding='same',name='layer5',pool_type='max')
    layer6 = conv(layer5, 128, (3, 3), padding='same', strides=(1, 1), name='layer6')
    layer7 = conv(layer6, 256, (3, 3), padding='same', strides=(1, 1), name='layer7')
    layer8 = pool(layer7, pool_size=(3, 3), strides=(2, 2), padding='same', name='layer8', pool_type='max')
    layer9 = conv(layer8, 256, (5, 5), padding='same', strides=(1, 1), name='layer9')
    x = Flatten()(layer9)
    x = Dense(64, activation='relu')(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)
    x = Model(data_input, x, name='mnist')
    return x

def cnn_model1(IMG_SIZE):
    data_input = Input(shape=(IMG_SIZE, IMG_SIZE,3))
    layer1 = conv(data_input, 32, (5, 5), padding='same', strides=(1, 1), name='layer1')
    x = fire_module(layer1,1,squeeze=16, expand=64,strides=2)
    x = BatchNormalization(axis=3, epsilon=1e-06)(x)
    x = fire_module(x,2, squeeze=32, expand=128)
    x = BatchNormalization(axis=3, epsilon=1e-06)(x)
    x = fire_module(x,4, squeeze=64, expand=256,strides=2)
    x = BatchNormalization(axis=3, epsilon=1e-06)(x)
    x = pool(x, pool_size=(3, 3), strides=(2, 2), padding='same', name='layer2', pool_type='max')
    x = fire_module(x, 5, squeeze=32, expand=128,strides=1)
    x = BatchNormalization(axis=3, epsilon=1e-06)(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)
    x = Model(data_input, x, name='mnist')
    return x

def cnn_model2(IMG_SIZE):
    data_input = Input(shape=(IMG_SIZE, IMG_SIZE,3))
    layer1 = conv(data_input, 32, (5, 5), padding='same', strides=(1, 1), name='layer1')
    x = fire_module(layer1, 1, squeeze=16, expand=64, strides=2)
    x = BatchNormalization(axis=3, epsilon=1e-06)(x)
    x = fire_module(x,2, squeeze=32, expand=128)
    x = BatchNormalization(axis=3, epsilon=1e-06)(x)
    x = pool(x, pool_size=(3, 3), strides=(2, 2), padding='same', name='layer2', pool_type='max')
    x = fire_module(x, 3, squeeze=16, expand=64,strides=1)
    x = BatchNormalization(axis=3, epsilon=1e-06)(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)
    x = Model(data_input, x, name='mnist')
    return x

def cnn_model3(IMG_SIZE):
    data_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = InvertedResidualBlock(data_input, expand=4, out_channels=32, repeats=1, stride=1, weight_decay=0.01,block_id=1)
    x = InvertedResidualBlock(x, expand=4, out_channels=64, repeats=2, stride=2, weight_decay=0.01,block_id=2)
    x = InvertedResidualBlock(x, expand=4, out_channels=128, repeats=2, stride=2, weight_decay=0.01,block_id=3)
    x = Flatten()(x)
    # x = Dense(16, activation='relu')(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)
    x = Model(data_input, x, name='mnist')
    return x
def fire_module(x, fire_id, squeeze=16, expand=64,strides =1):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"

    s_id = 'fire' + str(fire_id) + '/'

    x = Conv2D(squeeze, (1, 1), strides = (strides,strides),padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=3, name=s_id + 'concat')
    return x

def Relu6(x, **kwargs):
    return Activation(relu6, **kwargs)(x)

def InvertedResidualBlock(x, expand, out_channels, repeats, stride, weight_decay, block_id):
    '''
    This function defines a sequence of 1 or more identical layers, referring to Table 2 in the original paper.
    :param x: Input Keras tensor in (B, H, W, C_in)
    :param expand: expansion factor in bottlenect residual block
    :param out_channels: number of channels in the output tensor
    :param repeats: number of times to repeat the inverted residual blocks including the one that changes the dimensions.
    :param stride: stride for the 1x1 convolution
    :param weight_decay: hyperparameter for the l2 penalty
    :param block_id: as its name tells
    :return: Output tensor (B, H_new, W_new, out_channels)
    '''
    channel_axis = -1
    in_channels = K.int_shape(x)[channel_axis]
    x = Conv2D(expand * in_channels, 1, padding='same', strides=stride, use_bias=False,
                kernel_regularizer=l2(weight_decay), name='conv_%d_0' % block_id)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_1' % block_id)
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=1,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='conv_dw_%d_0' % block_id )(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_dw_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_2' % block_id)
    x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
               kernel_regularizer=l2(weight_decay), name='conv_bottleneck_%d_0' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_0_bn' % block_id)(x)

    for i in xrange(1, repeats):
        x1 = Conv2D(expand*out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay), name='conv_%d_%d' % (block_id, i))(x)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name='conv_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1,name='conv_%d_%d_act_1' % (block_id, i))
        x1 = DepthwiseConv2D((3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=1,
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            name='conv_dw_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9, name='conv_dw_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1, name='conv_dw_%d_%d_act_2' % (block_id, i))
        x1 = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay),name='conv_bottleneck_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_%d_bn' % (block_id, i))(x1)
        x = add([x, x1], name='block_%d_%d_output' % (block_id, i))
    return x