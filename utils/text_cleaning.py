"""
module for text cleaning
"""
import re

import numpy as np


def clean_plot_summary(plot_summary, to_print=False):
    """
    Returns a cleaned plot summary
    :param str plot_summary:
    :param bool to_print: Should erros be printed
    :return:
    """
    try:
        # remove '{{word}}' or '{{text>'
        x = re.sub(r'\s*{{.+?(\}{2}|\>)\s?', '', str(plot_summary))
        # remove links
        x = re.sub(r'http(s?)://\S*\s*', '', x)
        # remove punctuation
        x = re.sub(r'[\?\.\,\;\:\"\(\)\{\}\[\]\\\/]', ' ', x)
        # remove double space
        x = re.sub(r'\s+', ' ', x)
        # remove \' that is not in middle of a word)
        x = re.sub(r'\'(\w+)\'', r'\1', x)
        if not x:
            if to_print:
                print('Plot Summary empty after cleaning')
            return np.nan
        return x
    except Exception as e:
        if to_print:
            print(f'Failed with exception: {str(e)}')
            print('Plot Summary cannot be cleaned')
        return np.nan


def remove_stopwords(plot_summary, stopwords, stemmer):
    """
    Remove stopwords from plot summary
    :param list(str) plot_summary: tokenized plot summary
    :param set(str) stopwords:
    :param nltk.stem stemmer:
    :return:
    """
    return ' '.join([stemmer.stem(word) for word in plot_summary if word not in stopwords])
