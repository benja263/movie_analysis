import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


class GenresDataset(Dataset):
    def __init__(self, plot, genres, mapping, tokenizer, max_len):
        """

        :param plot:
        :param list(int) genres:
        :param mapping:
        :param tokenizer:
        :param max_len:
        """
        self.plot = plot
        self.genres = genres
        self.encoded_genres = encode_ids(genres, mapping)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.plot)

    def __getitem__(self, item):
        plot = self.plot[item]
        genres = self.genres[item]
        encoded_genres = self.encoded_genres[item]
        encoding = self.tokenizer.encode_plus(plot, add_special_tokens=True, max_length=self.max_len,
                                              return_token_type_ids=False, pad_to_max_length=True,
                                              return_attention_mask=True, return_tensors='pt', truncation=True)

        return {'plot': plot, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'genres': genres, 'encoded_genres': torch.tensor(encoded_genres, dtype=torch.float)}


def create_genres_data_loader(df, mapping, tokenizer, max_len, batch_size, plot_col, genre_col):
    """

    :param df:
    :param mapping:
    :param tokenizer:
    :param max_len:
    :param batch_size:
    :param plot_col:
    :param genre_col:
    :return:
    """
    df.loc[:, genre_col] = df[genre_col].apply(genre_yaml_to_list)
    dataset = GenresDataset(plot=df[plot_col].to_numpy(), genres=df[genre_col].to_numpy(), mapping=mapping,
                            tokenizer=tokenizer, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)


def encode_ids(data, mapping):
    """

    :param data:
    :param mapping:
    :return:
    """
    nb_genres = len(mapping)
    encoding_list = []
    for genre_list in data:
        encoding = np.zeros(nb_genres, dtype=int)
        for genre in genre_list:
            encoding[mapping[genre]] = 1
        encoding_list.append(encoding)
    return encoding_list


def genre_yaml_to_list(x):
    """

    :param x
    """
    x = yaml.load(x, Loader=yaml.BaseLoader)
    return list(x.values())

