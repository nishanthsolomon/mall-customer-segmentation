from sklearn.cluster import KMeans
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly as py
from plot_prediction import plot_results

def kmeans(data, K):
    X = data[[
        'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values
    km = KMeans(
        n_clusters=K, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(X)

    plot_results(data, y_km)
    


def distortions(data):
    X = data[[
        'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values
    # calculate distortion for a range of number of cluster
    distortions = []
    cluster = [4, 6, 8, 10]
    for c in cluster:
        km = KMeans(
            n_clusters=c, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(cluster, distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method to determine the Optimum cluster number')
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('/home/nishanth/gits/mall-customer-segmentation/dataset/dataset.csv')
    data.isnull().any().any()
    kmeans(data, 4)
    kmeans(data, 6)
    kmeans(data, 8)
    kmeans(data, 10)
    distortions(data)
