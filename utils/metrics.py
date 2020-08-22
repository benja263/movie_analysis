"""
Module containing methods for returning predicted vs target output evaluation metrics
"""
import torch
import numpy as np


def confusion_matrix(predictions, targets, is_torch=False):
    """
    Return confusion matrix
    :param predictions:
    :param targets:
    :param bool is_torch: are inputs torch arrays or numpy arrays
    :return:
    """
    library = {True: torch, False: np}
    tp = library[is_torch].sum((predictions == 1) & (targets == 1), dtype=library[is_torch].float)
    tn = library[is_torch].sum((predictions == 0) & (targets == 0), dtype=library[is_torch].float)
    fp = library[is_torch].sum((predictions == 1) & (targets == 0), dtype=library[is_torch].float)
    fn = library[is_torch].sum((predictions == 0) & (targets == 1), dtype=library[is_torch].float)
    return tp, tn, fp, fn


def classifcation_metrics(tp, tn, fp, fn):
    """
    Return classifcation metrics: precision, recall, accuracy and F1 score as a dictionary
    :param float tp: true positive
    :param float tn: true negative
    :param float fp: false positive
    :param float fn: false negative
    :return:
    """
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2.0 * precision * recall) / (recall + precision) if recall + precision > 0.0 else 0.0
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}
