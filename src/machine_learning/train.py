import pickle

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History

from src.models.vgg16 import create_model as vgg16
from src.models.efficientnet import EfficientNetB0 as efficientnet
from src.models.vit import VisionTransformer
from config import input_size, batch_size, path_to_train_set, path_to_valid_set, path_to_test_set, path_to_weights, path_to_history, img_format, \
    vit_settings, channels, num_classes


class Trainer:
    def __init__(self, train_gen, valid_gen, model_name):
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.input_size = input_size
        self.channels = channels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.vit_settings = vit_settings
        self.train_steps = pd.read_csv('{}/{}/{}'.format(path_to_train_set, img_format, 'train.csv')).shape[0] / batch_size
        self.valid_steps = pd.read_csv('{}/{}/{}'.format(path_to_valid_set, img_format, 'valid.csv')).shape[0] / batch_size
        self.test_steps = pd.read_csv('{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv')).shape[0] / batch_size
        self.path_to_weights = '{}/{}.hdf5'.format(path_to_weights, model_name)
        self.path_to_history = '{}/{}.pickle'.format(path_to_history, model_name)

        self.model = None
        if model_name == 'vgg16':
            self.model = vgg16(img_dim=(input_size, input_size, channels))

        elif model_name == 'efficientnet':
            self.model = efficientnet(input_shape=(input_size, input_size, channels))

        elif model_name == 'vit':
            self.model = VisionTransformer(image_size=self.input_size,
                                           patch_size=self.vit_settings['patch_size'],
                                           num_layers=self.vit_settings['num_layers'],
                                           num_classes=self.num_classes,
                                           d_model=self.vit_settings['d_model'],
                                           num_heads=self.vit_settings['num_heads'],
                                           mlp_dim=self.vit_settings['mlp_dim'],
                                           channels=self.channels,
                                           dropout=self.vit_settings['dropout']).build_VisionTransformer()

    def train(self):
        self.model.compile(loss=tf.keras.losses.binary_crossentropy,
                           optimizer=tfa.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
                           metrics=["accuracy"])
        self.model.summary()
        # self.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])

        history = History()
        callbacks = [history,
                     EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=1e-4),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=0, min_lr=1e-7, verbose=1),
                     ModelCheckpoint(filepath=self.path_to_weights, verbose=1, save_best_only=False,
                                     save_weights_only=True, mode='auto')]

        history = self.model.fit(self.train_gen, steps_per_epoch=self.train_steps,
                                 validation_data=self.valid_gen, validation_steps=self.test_steps, epochs=50,
                                 verbose=1, callbacks=callbacks)

        with open(self.path_to_history, 'wb') as f:
            pickle.dump(history.history, f)
