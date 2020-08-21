import re

import numpy as np
import yaml


def A_minus_intersection(A, B):
    """
    Returns set containing all values in A minus intersection(A, B)
    :param set A:
    :param set B:
    :return:
    """
    return A - A.intersection(B)


def passed_time(t):
    """

    :param t:
    :return:
    """
    if t < 60:
        return f'{t:.2f} seconds'
    if t // 60 < 60:
        return f'{t // 60} minutes and {passed_time(t % 60)}'
    return f'{t // 3600} hours {passed_time(t % 3600)}'


def save_data(path, filename, data):
    """

    :param path:
    :param filename:
    :param data:
    :return:
    """
    train_data, val_data, test_data = data
    save_name = path / filename.split('.')[0]
    train_data.to_csv(f'{save_name}_train_set.csv')
    val_data.to_csv(f'{save_name}_val_set.csv')
    test_data.to_csv(f'{save_name}_test_set.csv')


def append_history(history_dict, accuracy, precision, recall, f1, loss, metric_type):
    history_dict[f'{metric_type}_acc'].append(accuracy)
    history_dict[f'{metric_type}_prec'].append(precision)
    history_dict[f'{metric_type}_recall'].append(recall)
    history_dict[f'{metric_type}_f1'].append(f1)
    history_dict[f'{metric_type}_loss'].append(loss)
    return history_dict


def ensure_ending(filename):
    if '.' in filename:
        return True
    return False


def print_metrics(metrics):
    for metric, metric_value in metrics.items():
        print(f"{metric}: {metric_value:.5f}", end=' ')
    print('')


def clean_plot_summary(string):
    """

    :param row:
    :return:
    """
    try:
        # remove '{{word}}' or '{{text>'
        x = re.sub(r'\s*{{.+?(\}{2}|\>)\s?', '', str(string))
        # remove links
        x = re.sub(r'http(s?)://\S*\s*', '', x)
        # remove punctuation
        x = re.sub(r'[\?\.\,\;\:\"\(\)\{\}\[\]\\\/]', ' ', x)
        # remove double space
        x = re.sub(r'\s+', ' ', x)
        # remove \' that is not in middle of a word)
        x = re.sub(r'\'(\w+)\'', r'\1', x)
        if not x:
            return np.nan
        return x
    except:
        return np.nan


def list_from_yaml(list_representation):
    return yaml.load(list_representation, Loader=yaml.BaseLoader)
