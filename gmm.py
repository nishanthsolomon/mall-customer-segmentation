import scipy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from plot_prediction import plot_results
from sklearn.mixture import GaussianMixture
from dataset_reader import get_data


class GMM():

    def __init__(self):
        columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)',
                   'Gender_Female', 'Gender_Male']
        
        self.X = get_data(columns)
        self.X_vals = self.X.values

    def train_predict_gmm(self, num_clusters):
        gmm = GaussianMixture(n_components=num_clusters).fit(self.X_vals)
        labels = gmm.predict(self.X_vals)

        plot_results(self.X, labels)

        score = silhouette_score(self.X_vals, labels)

        return labels, score


if __name__ == "__main__":

    scores = []
    clusters = []

    gmm = GMM()

    for n in range(4, 12, 2):
        labels, score = gmm.train_predict_gmm(n)
        clusters.append(n)
        scores.append(score)

    fig = plt.figure()
    plt.plot(clusters, scores)
    fig.suptitle(
        'The Silhouette coefficient method for determining number of clusters', fontsize=20)
    plt.xlabel('Number of Clusters', fontsize=18)
    plt.ylabel('Silhouette Score', fontsize=16)
    fig.savefig('test.png')
    plt.show()
