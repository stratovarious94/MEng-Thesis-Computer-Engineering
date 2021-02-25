import pandas as pd
from sklearn.model_selection import train_test_split

from config import path_to_original_dataset, path_to_train_set, path_to_valid_set, path_to_test_set, img_format


class Preprocessor:
    """
    Pipeline that transforms the raw data into binarized split datasets
    """
    def __init__(self):
        # read the df and format its 2 columns accordingly
        self.data = pd.read_csv(path_to_original_dataset)

    def __multilabel_processing(self):
        """
        Convert the label column from list of strings to binarized list
        and insert the extensions in front of the image names column
        """
        # read the raw dataset
        self.data['image_name'] = self.data['image_name'].map(lambda x: '{}.{}'.format(x, img_format))
        self.data['tags'] = self.data['tags'].map(lambda x: x.split())

        # create a df with the same number of rows as the dataset filled with the name of the unique values in tags
        label_names = self.data['tags'].explode().unique().tolist()
        label_df = pd.DataFrame([label_names] * self.data.shape[0], columns=label_names)

        # binarize the labels according to if they exist for each image or not
        self.data = pd.concat([self.data, label_df], axis=1)
        self.data[['image_name'] + label_names] = self.data.apply(lambda x: pd.Series([x[0]] + [1 if label in x[1] else 0 for label in x[2:]]), axis=1)

    def __split_dataset(self):
        """
        Split the dataset 60/20/20 train/valid/test
        """
        self.train, _, self.valid, _ = train_test_split(self.data, self.data, test_size=0.4)
        self.valid, _, self.test, _ = train_test_split(self.valid, self.valid, test_size=0.5)

    def __save_datasets(self):
        """
        Save each split of the dataset into its specified directory
        """
        self.train.to_csv('{}/{}/{}'.format(path_to_train_set, img_format, 'train.csv'))
        self.valid.to_csv('{}/{}/{}'.format(path_to_valid_set, img_format, 'valid.csv'))
        self.test.to_csv('{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv'))

    def preprocessing_pipeline(self):
        """
        Executes the whole preprocessing pipeline in sequence
        """
        self.__multilabel_processing()
        self.__split_dataset()
        self.__save_datasets()
