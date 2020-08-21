"""
Module for local loading/saving
"""
import json
import pickle

import torch

from model.model import MultiGenreLabeler
from model.parameters import ModelParameters
from utils.helpers import ensure_ending


def load_json(path, filename):
    filename = filename if ensure_ending(filename) else f"{filename}.json"
    try:
        with open(path / filename, 'r') as json_file:
            data = json.load(json_file)
        print(f'Loaded file: {filename} successfully')
        return data
    except FileNotFoundError:
        print(f'File: {filename} not found')
        return dict()


def load_pickle(path, filename):
    filename = filename if ensure_ending(filename) else f"{filename}.pkl"
    try:
        with open(path / filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        print(f'Loaded file: {filename} successfully')
        return data
    except FileNotFoundError:
        print(f'File: {filename} not found')


def save_json(file, path, filename):
    filename = filename if ensure_ending(filename) else f"{filename}.json"
    print(f'Saving {filename} to: {path}')
    with open(path / filename, 'w') as json_file:
        json.dump(file, json_file)


def save_pickle(file, path, filename):
    filename = filename if ensure_ending(filename) else f"{filename}.pkl"
    print(f'Saving {filename} to: {path}')
    with open(path / filename, 'wb') as pickle_file:
        pickle.dump(file, pickle_file)


def save_model(model, path, filename):
    """

    :param model:
    :param path:
    :param filename:
    :return:
    """
    filename = filename if ensure_ending(filename) else f"{filename}.pth"
    torch.save(model.state_dict(), path / filename)


def load_model(path, filename, device):
    """
    Returns trained model and metadata. The function assumes that they are in the same file path and are named:
        metadata - <filename>_metadata.pkl
        model - <filename>.pth
    :param path:
    :param filename:
    :param device:
    :return:
    """
    print('--Loading metadata --')
    metadata_filename = f'{filename}_metadata.pkl'
    metadata = load_pickle(path, metadata_filename)
    print('Model trained with following parameters:')
    for param, param_value in metadata['parameters'].items():
        print(f'-- {param}: {param_value}')
    params = ModelParameters(**metadata['parameters'])
    print('--Loading model --')
    model = MultiGenreLabeler(params)
    filename = filename if ensure_ending(filename) else f"{filename}.pth"
    model.load_state_dict(torch.load(path / filename,
                                     map_location=torch.device(device)))
    model.to(device)
    return model, metadata