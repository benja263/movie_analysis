"""
Module for pre_processing original CMU data
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from utils.helpers import clean_plot_summary
from utils.serialization import save_json

import numpy as np
import pandas as pd

MOVIE_METADATA_COLUMNS = {0: 'wikipedia_movie_id', 1: 'freebase_movie_id', 2: 'movie_name', 3: 'movie_release_date',
                          4: 'movie_box_office_revenue', 5: 'movie_runtime', 6: 'movie_languages', 7: 'movie_countries',
                          8: 'movie_genres'}
MINIMUM_NUMBER_OF_APPEARANCE = 200


def preprocess_data(path):
    """
    Pre-process data from original CMU data to one csv containing:
    'movie_name', 'movie_genres', 'movie_plot'.
    Genres are converted to indices and are stored in a genre to index mapping as a json file
     'genre_mapping.json'.
    :param path:
    :return:
    """
    movie_data = load_data(path)
    print('Preprocessing movie data')
    movie_data, genre_mapping = pre_process_movie_data(movie_data)
    print_metadata(movie_data, genre_mapping)

    print('-- Saving --')
    save_plot_summaries_name = path / 'plot_summaries.json'
    print(f'Saving movie data as csv as : prepared_movie_data.csv')
    movie_data.to_csv(path / 'prepared_movie_data.csv')
    print(f'Saving unique genre to index mapping as: genre_mapping.json')
    save_json(genre_mapping, path, filename='genre_mapping')
    print(f'Saving movie plots as: plot_summaries.json')
    plot_summaries_to_json(path, movie_data)
    print('-' * 10)
    print('-- Saved --')


def load_data(path):
    """
    Returns movie metadata with plot summaries included
    :param Path path:
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
    movie_metadata.loc[:, 'plot_summary'] = movie_metadata['plot_summary'].apply(clean_plot_summary)
    movie_metadata = movie_metadata.loc[movie_metadata['plot_summary'] != 'nan']
    return movie_metadata.set_index('wikipedia_movie_id').dropna(subset=['plot_summary', 'movie_genres'])


def pre_process_movie_data(df):
    """
    Returns pre-processed movie data as a DataFrame and genre mapping such that:
    - DataFrame: index = movie name, columns = [plot summary, genres] where the genres are a dictionary with
                keys = order of genre appearance in a movie's genres and values = genres
    - genre mapping: keys = index of unique genre, values = unique genre
    :param pd.DataFrame df:
    :return:
    """

    movies_data = dict()
    unique_genre_count = defaultdict(lambda: 0)
    print('Finding unique genres')
    for row_number, (movie_id, row) in enumerate(df.iterrows(), start=0):
        movie_genres = json.loads(row['movie_genres'])
        genre_dict = np.nan
        if movie_genres:
            genre_values = list(movie_genres.values())
            for genre in genre_values:
                unique_genre_count[genre] += 1
        plot = row['plot_summary']
        movie_name = row['movie_name']
        movies_data[row_number] = {'movie_name': movie_name, 'plot_summary': plot, 'genres': genre_values}
    movies_data = pd.DataFrame.from_dict(movies_data, orient='index').dropna()
    print(f'Only keepnig genres that appear at least {MINIMUM_NUMBER_OF_APPEARANCE} times in {len(movies_data)} movies')
    accepted_genres = set([genre for genre, genre_count in unique_genre_count.items() if genre_count >= MINIMUM_NUMBER_OF_APPEARANCE])
    movies_data.loc[:, 'genres'] = movies_data['genres'].apply(lambda x: list(set(x).intersection(accepted_genres)))
    genre_mapping = {genre: ind for ind, genre in enumerate(accepted_genres)}
    return movies_data.set_index('movie_name'), genre_mapping


def print_metadata(df, mapping):
    """
    Print statistics of movie data
    """
    temp = df.copy()
    temp.loc[:, 'word_count'] = temp['plot_summary'].apply(lambda x: len(x.split()))
    temp.loc[:, 'genre_count'] = temp['genres'].apply(lambda x: len(x))
    word_stats, word_stats_quantiles = temp.word_count.describe(), temp.word_count.quantile(q=[0.9, 0.95, 0.99])
    genre_stats = temp.genre_count.describe()
    print('-' * 10)
    print('Statistics')
    print('-' * 10)
    print(f"{word_stats['count']:.0f} movies")
    print(f"Number of unique genres: {len(mapping)}")
    print(f"Word count per plot summary - min: {word_stats['min']:.0f}, mean: {word_stats['mean']:.2f},"
          f" std: {word_stats['std']:.2f}, median: {word_stats['50%']:.2f}, 90% : {word_stats_quantiles[0.90]:.0f}, "
          f"95% : {word_stats_quantiles[0.95]:.0f}, 99% : {word_stats_quantiles[0.99]:.0f},"
          f" max: {word_stats['max']:.0f}")
    print(f"Genre count per movie - min: {genre_stats['min']:.0f}, mean: {genre_stats['mean']:.2f}, "
          f"std: {genre_stats['std']:.2f}, median: {genre_stats['50%']:.0f},  max: {genre_stats['max']:.0f}")
    print('-' * 10)

def plot_summaries_to_json(path, df):
    """
    Save plot summaries as json
    :param Path path: output path
    :param pd.DataFrame df: dataframe containing plot summaries
    :return:
    """
    movie_json = dict()
    for index, row in df.iterrows():
        movie_json[index] = row['plot_summary']
    save_json(movie_json, path, filename='plot_summaries')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for preparing data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', help='Path to data directory', type=Path
                        , default=Path.cwd()/ 'data')
    args = parser.parse_args()
    preprocess_data(path=args.path)
