import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import input_size, batch_size, color_mode, path_to_img_directory, img_format, \
    path_to_train_set, path_to_valid_set, path_to_test_set


class ImageGenerator:
    def __init__(self):
        self.train_datagen = ImageDataGenerator(rescale=1./255., horizontal_flip=True, vertical_flip=True)
        self.valid_test_datagen = ImageDataGenerator(rescale=1./255.)
        self.target_size = (input_size, input_size)
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.path_to_img_directory = '{}/{}'.format(path_to_img_directory, img_format)
        self.path_to_train_set = '{}/{}/{}'.format(path_to_train_set, img_format, 'train.csv')
        self.path_to_valid_set = '{}/{}/{}'.format(path_to_valid_set, img_format, 'valid.csv')
        self.path_to_test_set = '{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv')
        self.label_names = pd.read_csv(self.path_to_train_set).columns[3:]

    def get_train_generator(self):
        return self.train_datagen.flow_from_dataframe(
            pd.read_csv(self.path_to_train_set),
            directory=self.path_to_img_directory,
            x_col='image_name',
            y_col=self.label_names,
            target_size=self.target_size,
            color_mode=self.color_mode,
            class_mode='raw',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            validate_filenames=True
        )

    def get_valid_generator(self):
        return self.valid_test_datagen.flow_from_dataframe(
            pd.read_csv(self.path_to_valid_set),
            directory=self.path_to_img_directory,
            x_col='image_name',
            y_col=self.label_names,
            target_size=self.target_size,
            color_mode=self.color_mode,
            class_mode='raw',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            validate_filenames=True
        )

    def get_test_generator(self):
        return self.valid_test_datagen.flow_from_dataframe(
            pd.read_csv(self.path_to_test_set),
            directory=self.path_to_img_directory,
            x_col='image_name',
            y_col=self.label_names,
            target_size=self.target_size,
            color_mode=self.color_mode,
            class_mode='raw',
            batch_size=self.batch_size,
            shuffle=False,
            seed=42,
            validate_filenames=True
        )
