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
def kmeans(data,K):
    #X=np.array(data)

    X=data[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
    km = KMeans(
        n_clusters=K, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(X)
    labels3 = y_km
    centroids3 = km.cluster_centers_
    #Plotting clusters
    data['label3'] =  labels3
    trace1 = go.Scatter3d(
        x= data['Age'],
        y= data['Spending Score (1-100)'],
        z= data['Annual Income (k$)'],
        mode='markers',
        marker=dict(
            color = data['label3'], 
            size= 3,
            line=dict(
                color= data['label3'],
                width= 12
            ),
            opacity=0.8
        )
    )
    data_ = [trace1]
    layout = go.Layout(
        title= 'Clusters For KMeans K=' + str(K),
        scene = dict(
                xaxis = dict(title  = 'Age'),
                yaxis = dict(title  = 'Spending Score'),
                zaxis = dict(title  = 'Annual Income')
            )
    )
    fig = go.Figure(data=data_, layout=layout)
    py.offline.iplot(fig)
def distortions(data):
    X=data[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
    # calculate distortion for a range of number of cluster
    distortions = []
    cluster=[4,6,8,10]
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
if __name__=="__main__":
    data = pd.read_csv('C:\Spring2020\ANC\Mid2Project\Mall_Customers.csv')
    data.isnull().any().any()
    kmeans(data,4)
    kmeans(data,6)
    kmeans(data,8)
    kmeans(data,10)
    distortions(data)


