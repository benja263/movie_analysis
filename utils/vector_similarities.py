"""
Module for calculating vector similarities
"""
import numpy as np
import torch


def cosine_similarity(a, b, is_tensor=False):
    if is_tensor:
        cosine = torch.dot(a, b) / (torch.norm(a, p=2)*(torch.norm(b, p=2)))
        return cosine.item()
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def dot_product(a, b, is_tensor=False):
    if is_tensor:
        return torch.dot(a, b).item()
    return np.dot(a, b)


def euclidean_distance(a, b, is_tensor=False):
    if is_tensor:
        return torch.dist(a, b, p=2).item()
    return np.linalg.norm(np.array(a)-np.array(b))
