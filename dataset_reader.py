import pandas as pd


def create_dataset():
    dataset = pd.read_csv('./dataset/kaggle_dataset.csv')
    df = dataset.copy()

    # Visualizing data
    print(df.head())

    # GEtting the insides of the data
    print(df.isnull().sum())
    print(df.describe())
    print(df.info())

    # Making  the independent variables matrix
    X = df.iloc[:, [3, 4]].values

    # One Hot Encoding the categorical data - Gender
    df = pd.get_dummies(df, columns=['Gender'], prefix=['Gender'])

    df.to_csv('./dataset/dataset.csv', sep=',')


def get_data(columns):
    data = pd.read_csv('./dataset/dataset.csv', names=columns, header=0)

    return data


def get_pca_data():
    pca_data = pd.read_csv('./dataset/pca_dataset.csv')

    return pca_data


if __name__ == "__main__":
    create_dataset()
