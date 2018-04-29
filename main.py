from dataloader import prepare_data,prepare_data1
from cnn_model import cnn_model0,cnn_model1,cnn_model2,cnn_model3
from keras import backend as K
import numpy as np
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau,EarlyStopping
from time import  time
from keras.preprocessing.image import ImageDataGenerator
import os
import PIL.Image
from keras.preprocessing.image import img_to_array, array_to_img
from skimage import io,transform
from  utils import  preprocess_img


IMG_SIZE = 30
modelName = 'model2_30.h5'
model = cnn_model2(IMG_SIZE)
lr = 0.004
batch_size = 10
epoch = 60
start = time()
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
X_train, y_train = prepare_data1(IMG_SIZE)
datagen = ImageDataGenerator(featurewise_center=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             shear_range=0.1,
                             rotation_range=0)
def train():


    checkpoint = ModelCheckpoint('./models_digital/'+modelName,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1
                                 )

    tb_counter = len([log for log in os.listdir(os.path.expanduser('./logs_digital/')) if 'classify' in log]) +1
    tensorboard = TensorBoard(log_dir=os.path.expanduser('./logs/') + modelName + '_' + str(tb_counter),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=40,
                            epochs=epoch,
                            callbacks=[checkpoint,tensorboard],
                            max_queue_size = 3
                           )
    print '**************************************'
    end = time()
    print (end - start)

def test():
    dir_path = '/data/sunsiyuan/2018learning/handWriting/240x320p'
    model.load_weights('./models_digital/'+modelName)
    count = 0
    total_time = 0
    print model.summary()
    for img_path in os.listdir(dir_path):
        img = os.path.join(dir_path, img_path)
        #print "the prediction of picture %s is  :"%(img_path)
        #boxes =[(20, 90, 60, 170),(100, 90, 140, 170),(180, 90, 220, 170),(260, 90, 300, 170)]
        boxes = [(28, 105, 47, 155), (111, 104, 130, 156), (193, 104, 214, 158)]
        flag = 0
        for box in boxes:
            image = io.imread(img)
            image = image[box[1]:box[3], box[0]:box[2]]
            image = transform.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image[np.newaxis, :, :, :]
            start = time()
            pred = model.predict(image)
            end = time() - start
            total_time = total_time + end
            #print np.argmax(pred[0][:])
            if (np.argmax(pred[0][:]) == int(img_path[flag])):
                count = count + 1
            flag = flag + 1
        #print "****************************************"
    print total_time
    print count

if __name__ == '__main__':

    config = tf.ConfigProto()

    #cpu
    # config.gpu_options.allow_growth = False
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    #gpu
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    K.set_session(tf.Session(config=config))

    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] =='test':
        test()

