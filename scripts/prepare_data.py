"""

"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_data(path, filename):
    """

    :param path:
    :param filename:
    :return:
    """
    movie_data, genre2id = get_text_and_genres(path, filename)
    print_metadata(movie_data, genre2id)

    print('-- Saving --')
    save_data_name = path / f"{filename.split('.')[0]}.csv"
    save_mapping_name = path / f"{filename.split('.')[0]}_mapping.json"
    print(f'Saving movie data as csv as : {save_data_name}')
    movie_data.to_csv(save_data_name)
    print(f'Saving movie genre id mapping as: {save_mapping_name}')
    with open(save_mapping_name, 'w') as json_file:
        json.dump(genre2id, json_file)
    print('-'*10)
    print('-- Saved --')


def get_text_and_genres(path, filename):
    """

    :param path:
    :param filename:
    :return:
    """
    genre2id = {'Unspecified': 0}
    genre_id = 1
    movies_data = dict()
    with open(path / filename) as j_file:
        for movie_ind, movie in enumerate(j_file, start=0):
            movie_info = json.loads(movie)
            genre_values = ['Unspecified']
            if 'genres' in movie_info.keys() and movie_info['genres']:
                genre_values = list(movie_info['genres'].values())
                genre_dict = {}
                for genre_ind, genre in enumerate(genre_values):
                    genre_dict[genre_ind] = genre
                    if genre not in genre2id.keys():
                        genre2id[genre] = genre_id
                        genre_id += 1
            plot = movie_info['plot_summary'] if 'plot_summary' in movie_info.keys() else 'Unspecified'
            title = movie_info['title'] if 'title' in movie_info.keys() else f'Unspecified_{movie_ind+1}'
            movies_data[movie_ind] = {'title': title, 'plot': plot, 'genres': genre_dict}
    movies_data = pd.DataFrame.from_dict(movies_data, orient='index')
    return movies_data.set_index('title'), genre2id


def print_metadata(df, mapping):
    """

    :param df:
    :param mapping:
    :return:
    """
    temp = df.copy()
    temp.loc[:, 'word_count'] = temp['plot'].apply(lambda x: len(x.split()))
    temp.loc[:, 'genre_count'] = temp['genres'].apply(lambda x: len(x))
    temp.loc[:, 'genres'] = temp['genres'].apply(lambda x: list(x.values()))
    word_stats, word_stats_quantiles = temp.word_count.describe(), temp.word_count.quantile(q=[0.9, 0.95, 0.99])
    genre_stats = temp.genre_count.describe()
    print('-'*10)
    print('Statistics')
    print('-'*10)
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
    for n in range(1, ngram+1):
        print(f'-- ngram: {n}/{ngram} -- ')
        print(temp.loc[temp.genre_count == n, 'genres'].value_counts().head(n=5).to_string())
    print('-'*10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for preparing data'")
    parser.add_argument('-p', '--path', help='Path to directory', type=Path
                        , required=True)
    parser.add_argument('-fn', '--filename', help='filename containing data (filename ending is also required): example:'
                                                  ' data.json', type=str, required=True)
    args = parser.parse_args()
    prepare_data(path=args.path, filename=args.filename)
