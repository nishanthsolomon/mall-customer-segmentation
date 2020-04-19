from sklearn.cluster import KMeans

from dataset_reader import get_data
from plot_prediction import *


class KMEANS():

    def __init__(self):
        columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)',
                   'Gender_Female', 'Gender_Male']

        self.X = get_data(columns)
        self.X_vals = self.X.values

    def train_predict_kmeans(self, K):
        km = KMeans(
            n_clusters=K, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        y_km = km.fit_predict(self.X_vals)
        distortions_score = km.inertia_
        plot_results(self.X, y_km)

        return distortions_score


if __name__ == "__main__":

    scores = []
    clusters = []

    kmeans = KMEANS()

    for n in range(4, 12, 2):
        score = kmeans.train_predict_kmeans(n)
        clusters.append(n)
        scores.append(score)

    title = 'Elbow Method to determine the Optimum cluster number'
    x_label = 'Number of Clusters'
    y_label = 'Distortion'

    plot_scores(clusters, scores, title, x_label, y_label)
