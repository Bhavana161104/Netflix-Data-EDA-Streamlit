import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("Data/netflix_titles.csv")

    df['date_added'] = pd.to_datetime(
        df['date_added'], errors='coerce'
    )

    df.drop_duplicates(inplace=True)
    return df


def plot_content_type(df):
    fig, ax = plt.subplots()
    df['type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Movies vs TV Shows")
    return fig


def plot_top_genres(df):
    genres = (
        df['listed_in']
        .dropna()
        .str.split(', ')
        .explode()
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots()
    genres.plot(kind='barh', ax=ax)
    ax.set_title("Top 10 Genres")
    return fig


def plot_release_year(df):
    fig, ax = plt.subplots()
    df['release_year'].value_counts().sort_index().plot(ax=ax)
    ax.set_title("Release Year Distribution")
    return fig
