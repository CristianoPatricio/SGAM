"""
Script to train an attribute classifier

cristiano.patricio@ubi.pt
"""
from keras.engine import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
import numpy as np
from keras.preprocessing import image
import glob
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras_vggface.vggface import VGGFace
import os
from tensorflow import keras
import time
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime


class LFWA():
    '''Wraps the LFWA dataset, allowing an easy way to:
         - Select the features of interest,
         - Split the dataset into 'training', 'test' or 'validation' partition.
    '''

    def __init__(self, main_folder='/home/cristianopatricio/Documents/Datasets/LFWA/LFWA+/', selected_features=None, drop_features=[]):
        self.main_folder = main_folder
        self.images_folder = os.path.join(main_folder, 'lfw-deepfunneled/')
        self.attributes_path = os.path.join(main_folder, 'list_attr_LFWA.csv')
        self.partition_path = os.path.join(main_folder, 'list_eval_partition.csv')
        self.selected_features = selected_features
        self.features_name = []
        self.__prepare(drop_features)

    def __prepare(self, drop_features):
        '''do some preprocessing before using the data: e.g. feature selection'''
        # attributes:
        if self.selected_features is None:
            self.attributes = pd.read_csv(self.attributes_path)
            self.num_features = 40
        else:
            self.num_features = len(self.selected_features)
            self.selected_features = self.selected_features.copy()
            self.selected_features.append('image_id')
            self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

        # remove unwanted features:
        for feature in drop_features:
            if feature in self.attributes:
                self.attributes = self.attributes.drop(feature, axis=1)
                self.num_features -= 1

        self.attributes.set_index('image_id', inplace=True)
        #self.attributes.replace(to_replace=-1, value=0, inplace=True)
        self.attributes['image_id'] = list(self.attributes.index)

        self.features_name = list(self.attributes.columns)[:-1]

        # load ideal partitioning:
        self.partition = pd.read_csv(self.partition_path)
        self.partition.set_index('image_id', inplace=True)

    def split(self, name='training', drop_zero=False):
        '''Returns the ['training', 'validation', 'test'] split of the dataset'''
        # select partition split:
        if name is 'training':
            to_drop = self.partition.where(lambda x: x != 0).dropna()
        elif name is 'validation':
            to_drop = self.partition.where(lambda x: x != 1).dropna()
        elif name is 'test':  # test
            to_drop = self.partition.where(lambda x: x != 2).dropna()
        else:
            raise ValueError('LFWA.split() => `name` must be one of [training, validation, test]')

        partition = self.partition.drop(index=to_drop.index)

        # join attributes with selected partition:
        joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

        if drop_zero is True:
            # select rows with all zeros values
            return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
        elif 0 <= drop_zero <= 1:
            zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
            zero = zero.sample(frac=drop_zero)
            return joint.drop(index=zero.index)

        return joint


def load_data(lfwa, batch_size=32):
    # augumentations for training set:
    train_datagen = ImageDataGenerator(rotation_range=20,
                                       rescale=1. / 255,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # only rescaling the validation set
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # get training and validation set:
    train_split = lfwa.split('training', drop_zero=False)
    valid_split = lfwa.split('validation', drop_zero=False)

    # data generators:
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_split,
        directory=lfwa.images_folder,
        x_col='image_id',
        y_col=lfwa.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other'
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_split,
        directory=lfwa.images_folder,
        x_col='image_id',
        y_col=lfwa.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other'
    )

    return train_generator, valid_generator


def train_model(train_generator, valid_generator, lfwa, batch_size=64, num_epochs=1, lr=0.01):
    from keras import backend as K

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # LOAD MODEL
        model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
        #model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights=None,
        #                pooling='avg')  # pooling: None, avg or max

        # Last conv layer is 'conv5_block3_3_conv' (7,7,2048)

        # input
        x = model.output

        # fully-connected layer
        x = Dense(1536, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        predictions = Dense(40, activation='sigmoid')(x)

        model = Model(inputs=model.input, outputs=predictions)
        print(model.summary())

        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adadelta(learning_rate=lr),
                      metrics='binary_accuracy')

    # setup checkpoint callback:
    save_path = "models"
    model_path = f"{save_path}/weights-FC{lfwa.num_features}-VGGFace" + "-{val_binary_accuracy:.2f}.hdf5"

    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_binary_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_binary_accuracy',
        factor=0.1,
        patience=5,
        min_lr=0.0001)

    # fitting:
    history = model.fit_generator(
        train_generator,
        epochs=num_epochs,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        max_queue_size=1,
        shuffle=True,
        callbacks=[checkpoint, reduce_lr],
        verbose=1)

    # Evaluate
    # get testset, and setup generator:
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = lfwa.split('test', drop_zero=False)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_set,
        directory=lfwa.images_folder,
        x_col='image_id',
        y_col=lfwa.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other')

    # evaluate model:
    score = model.evaluate_generator(
        test_generator,
        steps=len(test_generator),
        max_queue_size=1,
        verbose=1)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return history


def plot_results(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('results/results_vggface_'+str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))+'.png')


if __name__ == '__main__':
    lfwa = LFWA()
    train_generator, valid_generator = load_data(lfwa, batch_size=32)
    history = train_model(train_generator, valid_generator, lfwa, batch_size=32, num_epochs=100, lr=0.01)
    plot_results(history)
