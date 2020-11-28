import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

import pickle

from config import path_to_test_set, path_to_predictions, img_format


class Evaluator:
    def __init__(self, model_name):
        test_set = pd.read_csv('{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv'))
        self.true_labels = test_set[test_set.columns[3:]].values
        self.predicted_labels = []
        with open('{}/{}.pickle'.format(path_to_predictions, model_name), 'rb') as f:
            while True:
                try:
                    self.predicted_labels.append(pickle.load(f))
                except EOFError:
                    break

    def evaluate(self):
        score = fbeta_score(self.true_labels, (np.array(self.predicted_labels) > 0.2)[0].astype(int),
                            beta=2, average='samples')
        print(score)
