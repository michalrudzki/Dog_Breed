import numpy as np
import random as rn
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, AvgPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception

from keras.optimizers import Adam, SGD
from keras.regularizers import l2, l1, l1_l2
import math

def evaluation(model,test_generator,nb_test_samples,batch_size):
    evaluate = model.evaluate_generator(test_generator,\
                        steps=math.ceil(nb_test_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True)
    
    min_val_loss = 9999.0
    min_val_loss_i = 0
    stop_after_epoch = 0
    
    for i, val_loss in enumerate(model.history.history['val_loss']):
        if val_loss <= min_val_loss:
            min_val_loss_i = i
            min_val_loss = val_loss
        stop_after_epoch=i+1
            
    acc_train = model.history.history['acc'][min_val_loss_i]
    acc_val = model.history.history['val_acc'][min_val_loss_i]

    loss_train = model.history.history['loss'][min_val_loss_i]
    loss_val = model.history.history['val_loss'][min_val_loss_i]
    
    loss_test = evaluate[0] 
    acc_test = evaluate[1]
    return acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch

def arch_1(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(), 

        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

def arch_2(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(), 

        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]


def arch_3(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(), 
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

def arch_4(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        
        Flatten(), 
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

def arch_5(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(), 

        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]


def arch_6(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(), 
        
        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

def arch_7(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        
        Flatten(), 
        
        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

def arch_8(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(), 

        Dense(1024, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]


def arch_9(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(), 
        
        Dense(1024, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

def arch_10(image_shape, num_classes, callbacks, train_generator, nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer):
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=image_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        
        Flatten(), 
        
        Dense(1024, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]


def arch_TL_1(image_shape, num_classes, callbacks, train_generator, \
                     nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer, base_model):
    
    model = Sequential([
        base_model,

        Flatten(), 

        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]


def arch_TL_2(image_shape, num_classes, callbacks, train_generator, \
                     nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer, base_model):
    
    model = Sequential([
        base_model,

        Flatten(), 

        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]


def arch_TL_3(image_shape, num_classes, callbacks, train_generator, \
                     nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer, base_model):
    
    model = Sequential([
        base_model,

        Flatten(), 

        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

def TL_phase2(image_shape, num_classes, callbacks, train_generator, \
                     nb_train_samples,validation_generator,nb_validation_samples,\
          batch_size,test_generator,nb_test_samples,regularizer,optimizer, model):
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    model.fit_generator(train_generator,\
                        samples_per_epoch=nb_train_samples, \
                        epochs=100,\
                        validation_data=validation_generator, \
                        validation_steps=math.ceil(nb_validation_samples / batch_size),\
                        workers=16,\
                        use_multiprocessing=True,\
                        callbacks=callbacks)

    acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch = evaluation(model,test_generator,nb_test_samples,batch_size)
    return model, [acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, stop_after_epoch]

