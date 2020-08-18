"""

"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

MOVIE_METADATA_COLUMNS = {0: 'wikipedia_movie_id', 1: 'freebase_movie_id', 2: 'movie_name', 3: 'movie_release_date',
                          4: 'movie_box_office_revenue', 5: 'movie_runtime', 6: 'movie_languages', 7: 'movie_countries',
                          8: 'movie_genres'}


def prepare_data(path):
    """

    :param path:
    :return:
    """
    movie_data = load_data(path)
    movie_data, genre_mapping = get_text_and_genres(movie_data)
    print_metadata(movie_data, genre_mapping)

    print('-- Saving --')
    save_data_name = path / 'prepared_movie_data.csv'
    save_mapping_name = path / 'genre_mapping.json'
    save_plot_summaries_name = path / 'plot_summaries.json'
    print(f'Saving movie data as csv as : {save_data_name}')
    movie_data.to_csv(save_data_name)
    print(f'Saving movie genre id mapping as: {save_mapping_name}')
    with open(save_mapping_name, 'w') as json_file:
        json.dump(genre_mapping, json_file)
    print(f'Saving movie plots as: {save_plot_summaries_name}')
    plot_summaries_to_json(path, movie_data)
    print('-' * 10)
    print('-- Saved --')


def load_data(path):
    """

    :param path:
    :return:
    """
    # load movie metadata
    movie_metadata = pd.read_csv(path / 'movie.metadata.tsv', sep='\t', header=None).rename(
        columns=MOVIE_METADATA_COLUMNS)
    movie_metadata['wikipedia_movie_id'] = movie_metadata['wikipedia_movie_id'].astype(int)
    # load plot summaries
    movie_id_mapping = dict()
    with open(path / "plot_summaries.txt", 'r') as f:
        reader = csv.reader(f, dialect='excel-tab')
        for row in reader:
            try:
                movie_id_mapping[int(row[0])] = row[1]
            except Exception as e:
                pass
    movie_metadata.loc[:, 'plot_summary'] = movie_metadata['wikipedia_movie_id'].map(movie_id_mapping)
    return movie_metadata.set_index('wikipedia_movie_id').dropna(subset=['plot_summary', 'movie_genres'])


def get_text_and_genres(df):
    """

    :param path:
    :param filename:
    :return:
    """

    genre_id = 0
    movies_data, genre_mapping = dict(), dict()
    for movie_ind, (movie_id, row) in enumerate(df.iterrows(), start=0):
        movie_genres = json.loads(row['movie_genres'])
        genre_dict = np.nan
        if movie_genres:
            genre_values = list(movie_genres.values())
            genre_dict = dict()
            for genre_ind, genre in enumerate(genre_values):
                genre_dict[genre_ind] = genre
                if genre not in genre_mapping.keys():
                    genre_mapping[genre] = genre_id
                    genre_id += 1
        plot = row['plot_summary']
        movie_name = row['movie_name']
        movies_data[movie_ind] = {'movie_name': movie_name, 'plot_summary': plot, 'genres': genre_dict}
    movies_data = pd.DataFrame.from_dict(movies_data, orient='index').dropna()
    return movies_data.set_index('movie_name'), genre_mapping


def print_metadata(df, mapping):
    """

    :param df:
    :param mapping:
    :return:
    """
    temp = df.copy()
    temp.loc[:, 'word_count'] = temp['plot_summary'].apply(lambda x: len(x.split()))
    temp.loc[:, 'genre_count'] = temp['genres'].apply(lambda x: len(x))
    temp.loc[:, 'genres'] = temp['genres'].apply(lambda x: list(x.values()))
    word_stats, word_stats_quantiles = temp.word_count.describe(), temp.word_count.quantile(q=[0.9, 0.95, 0.99])
    genre_stats = temp.genre_count.describe()
    print('-' * 10)
    print('Statistics')
    print('-' * 10)
    print(f"{word_stats['count']:.0f} movies")
    print(f"Number of unique genres: {len(mapping)}")
    print(f"Word count - min: {word_stats['min']:.0f}, mean: {word_stats['mean']:.2f},"
          f" std: {word_stats['std']:.2f}, median: {word_stats['50%']:.2f}, 90% : {word_stats_quantiles[0.90]:.0f}, "
          f"95% : {word_stats_quantiles[0.95]:.0f}, 99% : {word_stats_quantiles[0.99]:.0f},"
          f" max: {word_stats['max']:.0f}")
    print(f"Genre count - min: {genre_stats['min']:.0f}, mean: {genre_stats['mean']:.2f}, "
          f"std: {genre_stats['std']:.2f}, median: {genre_stats['50%']:.0f},  max: {genre_stats['max']:.0f}")
    ngram = int(genre_stats['50%']) if int(genre_stats['50%']) < 5 else 5
    print(f'Top 5 genre combination up to ngram: {ngram}')
    for n in range(1, ngram + 1):
        print(f'-- ngram: {n}/{ngram} -- ')
        print(temp.loc[temp.genre_count == n, 'genres'].value_counts().head(n=5).to_string())
    print('-' * 10)

def plot_summaries_to_json(path, data):
    """

    :param path:
    :param data:
    :return:
    """
    movie_json = dict()
    for index, row in data.iterrows():
        movie_json[index] = row['plot_summary']
    with open(path / 'plot_summaries.json', 'w') as f:
        json.dump(movie_json, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for preparing data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', help='Path to data directory', type=Path
                        , default=Path.cwd()/ 'data')
    args = parser.parse_args()
    prepare_data(path=args.path)
