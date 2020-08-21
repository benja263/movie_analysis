import torch
import numpy as np

def binary_labeling(p, threshold, device):
    """

    :param p:
    :param threshold:
    :param device:
    :return:
    """
    res = torch.zeros(size=tuple(p.size()), dtype=torch.float, device=device, requires_grad=False)
    res[torch.sigmoid(p) >= threshold] = 1
    return res


def confusion_matrix(predictions, targets, is_torch=False):
    """

    :param predictions:
    :param targets:
    :param bool is_torch: are inputs torch arrays or numpy arrays
    :return:
    """
    library = {True: torch, False: np}
    TP = library[is_torch].sum((predictions == 1) & (targets == 1), dtype=library[is_torch].float)
    TN = library[is_torch].sum((predictions == 0) & (targets == 0), dtype=library[is_torch].float)
    FP = library[is_torch].sum((predictions == 1) & (targets == 0), dtype=library[is_torch].float)
    FN = library[is_torch].sum((predictions == 0) & (targets == 1), dtype=library[is_torch].float)
    return TP, TN, FP, FN


def classifcation_metrics(TP, TN, FP, FN):
    """

    :param TP:
    :param TN:
    :param FP:
    :param FN:
    :return:
    """
    precision = TP / (TP + FP) if (TP + FP) > 0.0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0.0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = (2.0 * precision * recall) / (recall + precision) if recall + precision > 0.0 else 0.0
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}
