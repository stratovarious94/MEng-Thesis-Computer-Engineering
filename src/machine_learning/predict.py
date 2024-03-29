import pickle

import pandas as pd

from config import input_size, channels, batch_size, path_to_test_set, path_to_weights, path_to_predictions, \
    vit_settings, num_classes, img_format
from src.models.vgg16 import create_model as vgg16
from src.models.resnet import ResNet
from src.models.densenet import DenseNet
from src.models.mobilenet import MobileNet
from src.models.efficientnet import EfficientNetB0 as efficientnet
from src.models.vit import VisionTransformer


class Predictor:
    """
    Generalized prediction framework for the models
    """
    def __init__(self, test_gen, model_name):
        """
        :param test_gen: test set's generator
        :param model_name: chosen model's name
        """
        self.test_gen = test_gen
        self.input_size = input_size
        self.channels = channels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.vit_settings = vit_settings
        self.test_steps = pd.read_csv('{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv')).shape[0] / batch_size
        self.path_to_weights = '{}/{}.hdf5'.format(path_to_weights, model_name)
        self.path_to_predictions = '{}/{}.pickle'.format(path_to_predictions, model_name)

        # initialize the chosen model by its name
        self.model = None
        if model_name == 'vgg16':
            self.model = vgg16(img_dim=(input_size, input_size, channels))

        if model_name == 'resnet':
            self.model = ResNet(img_dim=(input_size, input_size, channels), num_classes=num_classes).create()

        if model_name == 'densenet':
            self.model = DenseNet(img_dim=(input_size, input_size, channels), num_classes=num_classes).create()

        if model_name == 'mobilenet':
            self.model = MobileNet(img_dim=(input_size, input_size, channels), num_classes=num_classes).create()

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

    def predict(self):
        # load the respective model's saved weights
        self.model.load_weights(self.path_to_weights)

        # perform predictions on the test set
        predicted_labels = self.model.predict(self.test_gen, steps=self.test_steps)

        # save the predictions as pickle
        with open(self.path_to_predictions, 'wb') as f:
            pickle.dump(predicted_labels, f)
