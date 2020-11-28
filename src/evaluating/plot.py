import pickle
from itertools import compress

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from config import path_to_history, path_to_plots, path_to_test_set, path_to_predictions, img_format


import pandas as pd


class Plotter:
    def __init__(self, model_name):
        self.model_name = model_name
        test_set = pd.read_csv('{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv'))
        self.label_names = pd.read_csv('{}/{}/{}'.format(path_to_test_set, img_format, 'test.csv')).columns[3:]
        self.true_labels = test_set[test_set.columns[3:]].values
        self.predicted_labels = []
        with open('{}/{}.pickle'.format(path_to_predictions, model_name), 'rb') as f:
            while True:
                try:
                    self.predicted_labels.append(pickle.load(f))
                except EOFError:
                    break

        self.history = []
        with (open('{}/{}.pickle'.format(path_to_history, model_name), 'rb')) as f:
            while True:
                try:
                    self.history.append(pickle.load(f))
                except EOFError:
                    break

    def plot(self):
        # print(confusion_matrix(self.true_labels, (np.array(self.predicted_labels) > 0.2).astype(int).squeeze()))
        fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(7, 9))
        for i in range(20):
            if i < len(self.label_names):
                cm = confusion_matrix(self.true_labels[:, i], (np.array(self.predicted_labels) > 0.2).astype(int).squeeze()[:, i])
                sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt='g', ax=ax[i // 4, i % 4])
                ax[i // 4, i % 4].set_title(self.label_names[i], fontdict={'fontsize': 14})
                ax[i // 4, i % 4].invert_xaxis()
                ax[i // 4, i % 4].invert_yaxis()

            ax[i // 4, i % 4].spines['top'].set_visible(False)
            ax[i // 4, i % 4].spines['bottom'].set_visible(False)
            ax[i // 4, i % 4].spines['right'].set_visible(False)
            ax[i // 4, i % 4].spines['left'].set_visible(False)
            ax[i // 4, i % 4].tick_params(top=False, bottom=False, right=False, left=False)

        plt.suptitle(self.model_name.upper(), fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('{}/{}_absolute_heatmap.png'.format(path_to_plots, self.model_name))

        fig, ax = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(7, 9))
        for i in range(20):
            if i < len(self.label_names):
                cm = confusion_matrix(self.true_labels[:, i], (np.array(self.predicted_labels) > 0.2).astype(int).squeeze()[:, i]) / len(self.true_labels)
                sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt='.2%', ax=ax[i // 4, i % 4])
                ax[i // 4, i % 4].set_title(self.label_names[i], fontdict={'fontsize': 14})
                ax[i // 4, i % 4].invert_xaxis()
                ax[i // 4, i % 4].invert_yaxis()

            ax[i // 4, i % 4].spines['top'].set_visible(False)
            ax[i // 4, i % 4].spines['bottom'].set_visible(False)
            ax[i // 4, i % 4].spines['right'].set_visible(False)
            ax[i // 4, i % 4].spines['left'].set_visible(False)
            ax[i // 4, i % 4].tick_params(top=False, bottom=False, right=False, left=False)

        plt.suptitle(self.model_name.upper(), fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('{}/{}_percentage_heatmap.png'.format(path_to_plots, self.model_name))

        plt.plot(list(range(len(self.history[0]['loss']))), self.history[0]['loss'], color='skyblue', linewidth=1)
        plt.plot(list(range(len(self.history[0]['val_loss']))), self.history[0]['val_loss'], color='orange', linewidth=1)
        plt.legend(['Training set', 'Validation set'], loc='upper right')
        plt.title('Loss plot for model: {}'.format(self.model_name.replace('_', ' ').upper()), fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.xlabel('Epochs', fontdict={'fontsize': 12})
        plt.ylabel('Loss', fontdict={'fontsize': 12})
        plt.ylim(top=0.25)
        plt.grid()
        plt.savefig('{}/{}.png'.format(path_to_plots, self.model_name))
