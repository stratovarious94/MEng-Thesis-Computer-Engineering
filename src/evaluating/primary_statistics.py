import pickle

import numpy as np
from sklearn.metrics import accuracy_score

from config import path_to_test_set, path_to_predictions, img_format


import pandas as pd


class PrimaryStatistics:
    """
    Playground for producing sheets for analytics
    This module is independent and is not used by the rest of the project
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.test_set = pd.read_csv('../../{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv'))
        self.true_labels = self.test_set[self.test_set.columns[3:]].values
        self.predicted_labels = []
        with open('../../{}/{}.pickle'.format(path_to_predictions, model_name), 'rb') as f:
            while True:
                try:
                    self.predicted_labels.append(pickle.load(f))
                except EOFError:
                    break

    def produce(self):
        """
        Groups all test set entries and predicted set entries by their label array
        Writes to an excel sheet each distinct label list along with how many times it appears in the set
        and how many times it is predicted EXACTLY correctly
        """
        true_labels = self.test_set[self.test_set.columns[3:]].reset_index()
        primary_labels = true_labels[true_labels['primary'] == 1]
        grouped_primary_labels = primary_labels.groupby(list(primary_labels.columns[1:]))['index'].apply(list)
        group_number = grouped_primary_labels.count()

        print(str(len(primary_labels)) + " rows contain the 'primary' label out of the " + str(len(true_labels)) + " rows in the test set")
        print("Found " + str(group_number) + " label combinations that contain the 'primary' label")

        res = []
        for i in range(group_number):
            p_label_combo = grouped_primary_labels.iloc[[i]]
            group_labels = [label for label, b in zip(p_label_combo.index.names, list(p_label_combo.index.values[0])) if b]
            group_indices = p_label_combo.iloc[0]
            group_indices_length = len(group_indices)

            p_predicted_labels = p_label_combo.explode().reset_index().drop('index', axis=1).values
            p_true_labels = (np.array(self.predicted_labels) > 0.2)[0].astype(int)
            p_true_labels = p_true_labels[group_indices]

            score = accuracy_score(p_true_labels, p_predicted_labels, normalize=False)
            res.append([group_labels, group_indices_length, score])

        res_df = pd.DataFrame(res, columns=['label_combination', 'total', 'correct_predictions'])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(res_df)
        res_df.to_excel('primary_statistics.xlsx')

    def truth_pred_pairs(self):
        """
        Produces an excel sheet with true labels list - predicted labels list pairs
        """
        true = self.test_set[self.test_set.columns[3:]].values
        pred = (np.array(self.predicted_labels) > 0.2)[0].astype(int)

        res = []
        for i in range(len(true)):
            res.append([true[i], pred[i]])

        res_df = pd.DataFrame(res, columns=['true_labels', 'predicted_labels'])
        res_df.to_excel('true_pred_pairs_{}.xlsx'.format(self.model_name))


PrimaryStatistics("efficientnet").truth_pred_pairs()
