from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization,Dropout
from keras.layers.advanced_activations import LeakyReLU
from skimage import  transform

def conv(x, out_filter,kernel_size, padding='same',strides=(1,1),name=None,is_bn =True):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
        activate_name = name + '_activate'
    else:
        bn_name = None
        conv_name = None
        activate_name = None
    if is_bn:
        x = Conv2D(out_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
        x = BatchNormalization(axis=3,epsilon=1e-06,name=bn_name)(x)
        x = Activation('relu',name=activate_name)(x)
    else:
        x = Conv2D(out_filter,kernel_size,padding=padding,strides=strides, activation='relu',name=conv_name)(x)
        x = Activation('relu', name=activate_name)(x)
    return x

def pool(x,pool_size=(3,3),strides=(1,1),padding='same',pool_type='max',name=None):
    if pool_type == 'max':
        x = MaxPooling2D(pool_size=pool_size,strides=strides,padding=padding,name=name)(x)
        return x
    if pool_type =='avg':
        x = AveragePooling2D(pool_size=pool_size,strides=strides,padding=padding,name=name)(x)
        return x

def dropout(x,rate=0.4):
    x= Dropout(rate)(x)
    return x

def preprocess_img(image):
      # make test data more like mnist, the part of number is white

    IMG_SIZE = 70
    image = transform.resize(image, (IMG_SIZE, IMG_SIZE))
    for i in range(0, IMG_SIZE):
        for j in range(0, IMG_SIZE):
            if image[i][j] < 0.1537:
                image[i][j] = -1
            else:
                image[i][j] = 1
    #image = transform.rotate(image,10)
    return image
