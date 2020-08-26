"""
Script for calculating the N-most similar movies by calculating the similarity between movie plot summaries of a
given movie to other movies
"""
import argparse
from pathlib import Path

import torch

from movie_classifier import MovieClassifier
from utils.serialization import load_json, load_model


def get_similar_movies(path, model_filename, movie_name, plot_summary, N, similarity_type='cosine'):
    """
    Returns the N most similar movie
    :param Path path:
    :param str model_filename: name of trained model
    :param str movie_name:
    :param int N: number of top similar movies to return
    :param str similarity_type: method to calculate similarity between 2 vectors (options are: 'cosine', 'distance',
    'dot')
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, metadata = load_model(path, model_filename, device)
    embeddings = load_json(path, f"{metadata['model_type']}_embeddings.json")

    movie_classifier = MovieClassifier(model, metadata, device, embeddings)
    most_similar_movies = movie_classifier.get_n_most_similar(movie_name,plot_summary, N, similarity_type)
    for ind, (similar_movie_name, similarity) in enumerate(most_similar_movies.items(), start=1):
        print(f'{ind}) - {similar_movie_name}, similarity score: {similarity:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for calculating the N most similar movies to a given movie",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp', '--model_path', help='Path to bert_model directory', type=Path
                        , default=Path.cwd().parent / 'trained_models')
    parser.add_argument('-dp', '--data_path', help='Path to data', type=Path
                        , default=Path.cwd().parent / 'data')
    parser.add_argument('-fn', '--filename',
                        help='model filename', type=str, required=True)
    parser.add_argument('-mn', '--movie_name',
                        help='Name of movie. Note: needs to be between double quotes "" ', type=str, required=True)
    parser.add_argument('-n', '--N',
                        help='Number of most similar movies to retrieve', type=int, default=5)
    parser.add_argument('-ps', '--plot_summary',
                        help='Plot summary to predict genres from. Note -- plot summary needs to be between double'
                             ' quotes', type=str, required=True)
    parser.add_argument('-st', '--similarity_type',
                        help='Type of vector similarity function to use, options are: cosine distance, dot', type=str,
                        default='cosine', choices=['cosine', 'distance', 'dot'])
    args = parser.parse_args()
    print('-- Entered Arguments --')
    for arg in vars(args):
        print(f'- {arg}: {getattr(args, arg)}')
    get_similar_movies(path=args.model_path, model_filename=args.filename,
                       movie_name=args.movie_name, plot_summary=args.plot_summary, N=args.N,
                       similarity_type=args.similarity_type)
