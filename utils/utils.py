import json
import pickle

import numpy as np
import torch

from model.model import GenreClassifier
from model.parameters import ModelParameters


def A_minus_intersection(A, B):
    """
    Returns all values in A minus intersection(A, B)
    :param A:
    :param B:
    :return:
    """
    return A - A.intersection(B)


def load_json(path, filename):
    try:
        with open(path / filename, 'r') as json_file:
            data = json.load(json_file)
        print(f'Loaded file: {filename} successfully')
        return data
    except FileNotFoundError:
        print(f'File: {filename} not found')
        return dict()


def load_pickle(path, filename):
    try:
        with open(path / filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        print(f'Loaded file: {filename} successfully')
        return data
    except FileNotFoundError:
        print(f'File: {filename} not found')


def save_json(file, path, filename):
    print(f'saving {filename} to: {path}')
    with open(path / filename, 'w') as json_file:
        json.dump(file, json_file)


def save_pickle(file, path, filename):
    print(f'saving {filename} to: {path}')
    with open(path / filename, 'wb') as pickle_file:
        pickle.dump(file, pickle_file)


def cosine_similarity(a, b, is_tensor=False):
    if is_tensor:
        cosine = torch.dot(a, b) / (torch.norm(a, p=2)*(torch.norm(b, p=2)))
        return cosine.item()
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def load_model(path, filename, device):
    print('-- metadata --')
    metadata = load_pickle(path, f"{filename}_metadata.pkl")
    params = ModelParameters(save_path=path, model_name=metadata['parameters']['model_name'],
                             num_labels=len(metadata['genre_mapping']),
                             pre_trained_model_name=metadata['parameters']['pre_trained_model_name'],
                             dropout=0.3)
    print('-- model --')
    model = GenreClassifier(params)
    model.load_state_dict(torch.load(path / f"{metadata['parameters']['model_name']}.pth",
                                     map_location=torch.device(device)))
    model.to(device)
    return model, metadata
