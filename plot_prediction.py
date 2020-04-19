import plotly.graph_objs as go
import plotly as py


def plot_results(data, labels):
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
        title='Clusters K = ' + str(len(set(labels))),
        scene=dict(
            xaxis=dict(title='Age'),
            yaxis=dict(title='Spending Score'),
            zaxis=dict(title='Annual Income')
        )
    )
    fig = go.Figure(data=data_, layout=layout)
    py.offline.plot(fig)
