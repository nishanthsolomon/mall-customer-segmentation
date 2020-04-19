from kmeans import KMEANS
from gmm import GMM


if __name__ == "__main__":
    num_clusters = int(input('Enter the number of clusters : '))

    kmeans = KMEANS()
    gmm = GMM()

    distortions_score = kmeans.train_predict_kmeans(num_clusters)
    print('Distortion score for kmeans clustering = ' + str(distortions_score))

    silhouette_score = gmm.train_predict_gmm(num_clusters)
    print('Silhouette score for gmm clustering = ' + str(silhouette_score))
