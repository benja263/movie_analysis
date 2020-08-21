"""
Module for converting movie information data from csv to pytorch Dataset suitable for training a pytorch model
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.helpers import list_from_yaml


class GenresDataset(Dataset):
    def __init__(self, plot_summaries, genres, mapping, tokenizer, max_len):
        """

        :param plot_summaries:
        :param list(int) genres:
        :param mapping:
        :param tokenizer:
        :param max_len:
        """
        self.plot_summaries = plot_summaries
        self.genres = genres
        self.encoded_genres = encode_ids(genres, mapping)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.plot_summaries)

    def __getitem__(self, item):
        plot_summary = self.plot_summaries[item]
        genres = self.genres[item]
        encoded_genres = self.encoded_genres[item]
        encoding = self.tokenizer.encode_plus(plot_summary, add_special_tokens=True, max_length=self.max_len,
                                              return_token_type_ids=False, pad_to_max_length=True,
                                              return_attention_mask=True, return_tensors='pt', truncation=True)

        return {'plot_summary': plot_summary, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'genres': genres, 'encoded_genres': torch.tensor(encoded_genres, dtype=torch.float)}


def create_genres_data_loader(df, mapping, tokenizer, max_len, batch_size, plot_col, genre_col, num_workers):
    """

    :param df:
    :param mapping:
    :param tokenizer:
    :param max_len:
    :param batch_size:
    :param plot_col:
    :param genre_col:
    :param num_workers:
    :return:
    """
    df.loc[:, genre_col] = df[genre_col].apply(list_from_yaml)
    dataset = GenresDataset(plot_summaries=df[plot_col].to_numpy(), genres=df[genre_col].to_numpy(), mapping=mapping,
                            tokenizer=tokenizer, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def encode_ids(genres_lists, mapping):
    """
    Returns 1-hot encodings of genres
    :param list(str) genres_lists: list of genres
    :param dict mapping: mapping between genre to index
    :return:
    """
    nb_genres = len(mapping)
    encoding_list = []
    for genres_list in genres_lists:
        encoding = np.zeros(nb_genres, dtype=int)
        for genre in genres_list:
            encoding[mapping[genre]] = 1
        encoding_list.append(encoding)
    return encoding_list


