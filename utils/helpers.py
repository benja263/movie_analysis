"""
Module containing helper functions
"""

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
    Return string representing passed time in seconds/minutes/hours depending on the number of seconds passed
    :param float t: time in seconds
    :return:
    """
    if t < 60:
        return f'{t:.2f} seconds'
    if t // 60 < 60:
        return f'{t // 60} minutes and {passed_time(t % 60)}'
    return f'{t // 3600} hours {passed_time(t % 3600)}'


def save_data(path, filename, data):
    """
    save to local training/validation/test pd.Dataframes as csv
    :param Path path:
    :param str filename:
    :param tuple(pd.DataFrame) data:
    :return:
    """
    train_data, val_data, test_data = data
    save_name = path / filename.split('.')[0]
    train_data.to_csv(f'{save_name}_train_set.csv')
    val_data.to_csv(f'{save_name}_val_set.csv')
    test_data.to_csv(f'{save_name}_test_set.csv')


def ensure_ending(filename):
    """
    Returns True if a filename has a file extension identifier
    :param str filename:
    :return:
    """
    if '.' in filename:
        return True
    return False


def list_from_yaml(list_representation):
    """
    Convert a string representation of a list to a list
    :param str list_representation:
    :return:
    """
    return yaml.load(list_representation, Loader=yaml.BaseLoader)
