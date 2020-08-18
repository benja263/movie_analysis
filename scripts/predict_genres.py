"""

"""
import argparse
import pickle
from pathlib import Path

import torch

from model.model import GenreClassifier
from model.parameters import ModelParameters
from movie_classifier import MovieClassifier


def predict_genres(path, filename, plot_summary):
    """

    :param path:
    :param filename:
    :param plot_summary:
    :return:
    """
    with open(path / f'{filename}_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    params = ModelParameters(**metadata['parameters'])
    model = GenreClassifier(params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    movie_classifier = MovieClassifier(model, metadata, device)
    prediction = movie_classifier.predict_genre_by_plot(plot_summary)
    print('-- Likely genres --')
    for ind, genre in enumerate(prediction.keys(), start=1):
        probability = prediction[genre]
        if probability > 0.5:
            print(f'{ind}) {genre}: {100.0*probability:.2f}')
    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for predicting movie genres",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', help='Path to trained model and metadata', type=Path
                        , default=Path.cwd().parent / 'trained_models')
    parser.add_argument('-fn', '--filename',
                        help='Filename of trained model, assumption is that model is saved as <filename>.pth and'
                             'metadata as <filename>_metadata.pkl', type=str, default='genre_classifier')
    parser.add_argument('-ps', '--plot_summary',
                        help='Plot summary to predict genres from. Note -- plot summary needs to be between double'
                             ' quotes', type=str, required=True)
    args = parser.parse_args()
    plot_summary = ' '.join(args.plot_summary)
    _ = predict_genres(args.path, args.filename, plot_summary)

