import plotly.graph_objs as go
import plotly as py
import matplotlib.pyplot as plt


def plot_results(data, labels, clustering_algorithm):
    data['label3'] = labels
    trace1 = go.Scatter3d(
        x=data['Age'],
        y=data['Spending Score (1-100)'],
        z=data['Annual Income (k$)'],
        mode='markers',
        marker=dict(
            color=data['label3'],
            size=5,
            line=dict(
                color=data['label3'],
                width=12
            ),
            opacity=0.8
        )
    )
    data_ = [trace1]
    layout = go.Layout(
        title=clustering_algorithm + ' Clusters K = ' + str(len(set(labels))),
        scene=dict(
            xaxis=dict(title='Age'),
            yaxis=dict(title='Spending Score'),
            zaxis=dict(title='Annual Income')
        )
    )
    fig = go.Figure(data=data_, layout=layout)

    py.offline.plot(fig)


def plot_scores(clusters, scores, title, x_label, y_label):
    fig = plt.figure()
    plt.plot(clusters, scores)
    fig.suptitle(title, fontsize=20)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    plt.show()
