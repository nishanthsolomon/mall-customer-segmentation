from sklearn.metrics import silhouette_score
from plot_prediction import *
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

        plot_results(self.X, labels, GMM.__name__)

        silhouette_score_ = silhouette_score(self.X_vals, labels)

        return silhouette_score_


if __name__ == "__main__":

    scores = []
    clusters = []

    gmm = GMM()

    for n in range(4, 12, 2):
        score = gmm.train_predict_gmm(n)
        clusters.append(n)
        scores.append(score)

    title = 'The Silhouette coefficient method for determining number of clusters'
    x_label = 'Number of Clusters'
    y_label = 'Silhouette Score'

    plot_scores(clusters, scores, title, x_label, y_label)
