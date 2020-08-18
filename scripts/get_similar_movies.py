"""

"""
import argparse
from pathlib import Path

import torch

from movie_classifier import MovieClassifier
from utils.helpers import load_json, load_model


def get_similar_movies(model_path, data_path, model_filename, movie_name, N, similarity_type='cosine'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, metadata = load_model(model_path, model_filename, device)
    embeddings = load_json(data_path, 'embeddings.json')

    movie_classifier = MovieClassifier(model, metadata, device, embeddings)
    most_similar_movies = movie_classifier.get_n_most_similar(movie_name, N, similarity_type)
    for ind, (similar_movie_name, similarity) in enumerate(most_similar_movies.items(), start=1):
        print(f'{ind}) - {similar_movie_name}, similarity score: {similarity:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for calculating the N most similar movies to a given movie",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp', '--model_path', help='Path to model directory', type=Path
                        , default=Path.cwd().parent / 'trained_models')
    parser.add_argument('-dp', '--data_path', help='Path to data', type=Path
                        , default=Path.cwd().parent / 'data')
    parser.add_argument('-fn', '--filename',
                        help='model filename', type=str, default='genre_classifier')
    parser.add_argument('-mn', '--movie_name',
                        help='Name of movie. Note: needs to be between double quotes "" ', type=str, required=True)
    parser.add_argument('-n', '--N',
                        help='Number of most similar movies to retrieve', type=int, default=5)
    parser.add_argument('-st', '--similarity_type',
                        help='Type of vector similarity function to use, options are: cosine distance, dot', type=str,
                        default='cosine', choices=['cosine', 'distance', 'dot'])
    args = parser.parse_args()
    print('-- Entered Arguments --')
    for arg in vars(args):
        print(f'- {arg}: {getattr(args, arg)}')
    get_similar_movies(model_path=args.model_path, data_path=args.data_path, model_filename=args.filename,
                       movie_name=args.movie_name, N=args.N, similarity_type=args.similarity_type)
