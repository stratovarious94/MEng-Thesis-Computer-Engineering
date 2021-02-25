import pickle

import numpy as np
import pandas as pd
import cv2
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from tensorflow.keras.models import Model

from config import path_to_test_set, path_to_weights, path_to_predictions, path_to_img_directory, path_to_explainable, \
    path_to_train_set, explainable_cnn_settings, num_classes, vit_settings
from src.models.vgg16 import create_model as vgg16
from src.models.resnet import ResNet
from src.models.densenet import DenseNet
from src.models.mobilenet import MobileNet
from src.models.efficientnet import EfficientNetB0 as efficientnet
from src.models.vit import VisionTransformer


class Explainer:
    """
    Attempts to 'peak' inside the network in order to explain the results in the context of Explainable AI
    For this purpose it uses the Grad-CAM technique and the last layer's tensor
    """
    def __init__(self, model_name):
        tf.compat.v1.disable_eager_execution()
        self.ex_channels = explainable_cnn_settings[model_name]['channels']
        self.ex_input_size = explainable_cnn_settings[model_name]['input_size']
        self.ex_format = explainable_cnn_settings[model_name]['ex_format']
        self.ex_last_conv_layer_name1 = explainable_cnn_settings[model_name]['last_conv_layer_name1']
        self.ex_last_conv_layer_name2 = explainable_cnn_settings[model_name]['last_conv_layer_name2']
        self.ex_last_conv_layer_filter_number = explainable_cnn_settings[model_name]['last_conv_layer_filter_number']

        test_set = pd.read_csv('{}/{}/{}'.format(path_to_test_set, self.ex_format, 'test.csv'))
        self.image_names = test_set['image_name']
        self.true_labels = test_set[test_set.columns[3:]].values
        self.label_names = pd.read_csv('{}/{}/{}'.format(path_to_train_set, self.ex_format, 'train.csv')).columns[3:]
        self.model_name = model_name
        self.path_to_weights = '{}/{}.hdf5'.format(path_to_weights, model_name)
        self.predicted_labels = []
        with open('{}/{}.pickle'.format(path_to_predictions, model_name), 'rb') as f:
            while True:
                try:
                    self.predicted_labels.append(pickle.load(f))
                except EOFError:
                    break

    def build_model(self):
        """
        Builds one of the available trained models
        :return: Model
        """
        model = None
        if self.model_name == 'vgg16':
            model = vgg16(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels))
        elif self.model_name == 'resnet':
            model = ResNet(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels), num_classes=num_classes).create()
        elif self.model_name == 'densenet':
            model = DenseNet(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels), num_classes=num_classes).create()
        elif self.model_name == 'mobilenet':
            model = MobileNet(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels), num_classes=num_classes).create()
        elif self.model_name == 'efficientnet':
            model = efficientnet(input_shape=(self.ex_input_size, self.ex_input_size, self.ex_channels))
        elif self.model_name == 'vit':
            model = VisionTransformer(image_size=self.ex_input_size,
                                      patch_size=vit_settings['patch_size'],
                                      num_layers=vit_settings['num_layers'],
                                      num_classes=num_classes,
                                      d_model=vit_settings['d_model'],
                                      num_heads=vit_settings['num_heads'],
                                      mlp_dim=vit_settings['mlp_dim'],
                                      channels=self.ex_channels,
                                      dropout=vit_settings['dropout']).build_VisionTransformer()
        model.load_weights(self.path_to_weights)
        model.summary()
        return model

    def build_cut_model(self):
        """
        Builds one of the available trained models all the way until the last convolution layer
        :return: Model
        """
        model = None
        if self.model_name == 'vgg16':
            model = vgg16(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels))
        elif self.model_name == 'resnet':
            model = ResNet(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels), num_classes=num_classes).create()
        elif self.model_name == 'densenet':
            model = DenseNet(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels), num_classes=num_classes).create()
        elif self.model_name == 'mobilenet':
            model = MobileNet(img_dim=(self.ex_input_size, self.ex_input_size, self.ex_channels), num_classes=num_classes).create()
        elif self.model_name == 'efficientnet':
            model = efficientnet(input_shape=(self.ex_input_size, self.ex_input_size, self.ex_channels))
        elif self.model_name == 'vit':
            model = VisionTransformer(image_size=self.ex_input_size,
                                      patch_size=vit_settings['patch_size'],
                                      num_layers=vit_settings['num_layers'],
                                      num_classes=num_classes,
                                      d_model=vit_settings['d_model'],
                                      num_heads=vit_settings['num_heads'],
                                      mlp_dim=vit_settings['mlp_dim'],
                                      channels=self.ex_channels,
                                      dropout=vit_settings['dropout']).build_VisionTransformer()
        model.load_weights(self.path_to_weights)
        model = Model(model.input, model.get_layer(self.ex_last_conv_layer_name2).output)
        model.summary()
        return model

    def explain(self):
        """
        Generate the Grad-CAM and last conv layer images
        """
        # build the 2 versions of the model
        model = self.build_model()
        last_conv_model = self.build_cut_model()

        for i, label_name in enumerate(self.label_names):
            # This is the algorithm for the last convolution layer's tensor image
            # Get the index of the image that was classified correctly with the most confidence for the class
            predicted_col_proba = np.array(self.predicted_labels)[0][:, i]
            predicted_col_argsort = predicted_col_proba.argsort()[::-1]
            predicted_col = (predicted_col_proba > 0.2).astype(int)
            true_col = self.true_labels[:, 0]

            representative_image_index = None
            for most_probable_arg_index in predicted_col_argsort:
                if predicted_col[most_probable_arg_index] == true_col[most_probable_arg_index]:
                    representative_image_index = most_probable_arg_index
                    break

            # Resize the image to fit the neural network and keep the original resized image
            original_img = io.imread('{}/{}/{}'.format(path_to_img_directory, self.ex_format, np.array(self.image_names)[representative_image_index]))
            original_img = cv2.normalize(original_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            original_img = cv2.resize(original_img, dsize=(self.ex_input_size, self.ex_input_size), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(original_img, axis=0)
            original_img = original_img[:, :, :3]

            # Get the output of the neural network for this image as a tensor
            model.predict(np.array(img))
            class_output = model.output[:, i]
            last_conv_layer = model.get_layer(self.ex_last_conv_layer_name1).output
            # if self.model_name == 'vit':
            #     last_conv_layer = tf.nn.relu(tf.reshape(last_conv_layer[:, :256, :], (-1, 16, 16, 1024)))

            # Get the output for the cut model
            cut_img = last_conv_model.predict(np.array(img))[0]
            if self.model_name == 'vit':
                cut_img = np.reshape(cut_img[:256, :], (16, 16, 1024))
            cut_img = np.mean(cut_img, axis=-1)
            cut_img = cv2.normalize(cut_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if self.model_name == 'vit':
                cut_img[0, 0] = np.mean(cut_img)
                cut_img = cv2.normalize(cut_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cut_img = cv2.resize(cut_img, (self.ex_input_size, self.ex_input_size))

            # This is the algorithm of the Grad-CAM model
            # Refine the output of the last convolutional layer according to the class output
            grads = K.gradients(class_output, last_conv_layer)[0]
            if self.model_name == 'vit':
                last_conv_layer = tf.reshape(last_conv_layer[:, :256, :], (-1, 16, 16, 1024))
                last_conv_layer = last_conv_layer / tf.norm(last_conv_layer)

                grads = tf.reshape(grads[:, :256, :], (-1, 16, 16, 1024))
                grads = grads / tf.norm(grads)

            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            iterate = K.function([model.input], [pooled_grads, last_conv_layer[0]])
            pooled_grads_value, conv_layer_output_value = iterate([img])
            for j in range(self.ex_last_conv_layer_filter_number):
                conv_layer_output_value[:, :, j] *= pooled_grads_value[j]

            # Create a 16x16 heatmap and scale it to the same size as the original image
            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap, (self.ex_input_size, self.ex_input_size))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap, 0.4, 0)

            # save the original image
            plt.matshow(original_img)
            plt.axis('off')
            plt.title(label_name, fontdict={'fontsize': 18})
            plt.savefig('{}/{}/{}_{}.png'.format(path_to_explainable, 'original', self.model_name, label_name), bbox_inches='tight', pad_inches=0.1)

            # save the cut image
            plt.matshow(cut_img, cmap=plt.get_cmap('Spectral'))
            plt.colorbar(shrink=0.75, ticks=np.linspace(0, 1, 11).tolist())
            plt.axis('off')
            plt.title(label_name, fontdict={'fontsize': 18})
            plt.savefig('{}/{}/{}_{}.png'.format(path_to_explainable, 'cut', self.model_name, label_name), bbox_inches='tight', pad_inches=0.1)

            # save the superimposed gradcam image
            plt.matshow(superimposed_img, cmap=plt.get_cmap('Spectral'))
            plt.colorbar(shrink=0.75, ticks=np.linspace(0, 1, 11).tolist())
            plt.axis('off')
            plt.title(label_name, fontdict={'fontsize': 18})
            plt.savefig('{}/{}/{}_{}.png'.format(path_to_explainable, 'gradcam', self.model_name, label_name), bbox_inches='tight', pad_inches=0.1)
