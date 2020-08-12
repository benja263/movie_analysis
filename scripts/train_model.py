import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
import time


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

from model.genres_dataset import create_genres_data_loader
from model.model import GenreClassifier
from model.parameters import ModelParameters


def train_model(path, filename, mapping_filename, params, load_model):
    """

    :param path:
    :param filename:
    :param mapping_filename:
    :param params:
    :param load_model:
    :return:
    """
    print('-- Loading Data --')
    data = pd.read_csv(path / filename)
    with open(path / mapping_filename, 'r') as json_file:
        mapping = json.load(json_file)
    print('-- Splitting Data --')
    print(f'train data ratio: {params.train_split},'
          f' validation_ratio: {((1 - params.train_split) * params.test_split):.2f}, '
          f'test_ratio: {((1.0 - params.train_split) * (1.0 - params.test_split)):.2f} ')
    train_data, test_data = train_test_split(data, train_size=params.train_split, shuffle=True,
                                             random_state=params.random_state)
    val_data, test_data = train_test_split(test_data, train_size=params.test_split, shuffle=True,
                                           random_state=params.random_state)
    print(f'train data has {len(train_data)} movies,'
          f' validation data has {len(val_data)} movies,'
          f' test data has {len(test_data)} movies,')
    data = (train_data, val_data, test_data)
    print('-- Saving Data -- ')
    save_data(path, filename, data)

    print('-- Generating Data Loaders -- ')
    tokenizer = BertTokenizer.from_pretrained(params.pre_trained_model_name)
    data_loaders = get_data_loaders(data, mapping, tokenizer, params)
    params.num_labels = len(mapping)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')
    print('-- Training Model -- ')
    model = GenreClassifier(params)
    if load_model:
        print('-- Loading Model --')
        model.load_state_dict(torch.load(params.save_path / f'{params.model_name}.pth',
                                         map_location=torch.device(device)))
    num_labels = len(mapping)
    training(model, data_loaders, params, num_labels, load_model, device)


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


def get_data_loaders(data, mapping, tokenizer, params):
    """

    :param data:
    :param mapping:
    :param tokenizer:
    :param params:
    :return:
    """
    train_data, val_data, test_data = data
    train_data_loader = create_genres_data_loader(train_data, mapping, tokenizer, params.max_encoding_length,
                                                  params.batch_size, plot_col='plot_summary', genre_col='genres')
    val_data_loader = create_genres_data_loader(val_data, mapping, tokenizer, params.max_encoding_length,
                                                params.batch_size, plot_col='plot_summary', genre_col='genres')
    test_data_loader = create_genres_data_loader(test_data, mapping, tokenizer, params.max_encoding_length,
                                                 params.batch_size, plot_col='plot_summary', genre_col='genres')
    return {'train': train_data_loader, 'validation': val_data_loader, 'test': test_data_loader}


def training(model, data_loader, params, num_labels, load_model, device):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                   num_training_steps=len(data_loader['train']) * params.n_epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    model_info = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'device': device,
                  'loss_fn': loss_fn}

    history = defaultdict(list)
    start_epoch = 1
    if load_model:
        with open(params.save_path / f'{params.model_name}_model_history.pkl', 'rb') as f:
            history = pickle.load(f)
            start_epoch = history['epoch'][-1]
    best_acc = 0.0
    start_time = time.time()
    for i in range(start_epoch, start_epoch + params.n_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {i}/{params.n_epochs}')
        print('-' * 10)
        tr_acc, tr_loss = train_epoch(**model_info, data_loader=data_loader['train'], num_labels=num_labels)
        val_acc, val_loss = eval_model(model, data_loader['validation'], loss_fn, device, num_labels)

        history['epoch'].append(i)
        history['tr_acc'].append(tr_acc)
        history['tr_loss'].append(tr_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        print(f'training accuracy: {tr_acc:.2f}, training loss: {tr_loss:.2f}')
        print(f'validation accuracy: {val_acc:.2f}, validation loss: {val_loss:.2f}')
        print('-' * 10)
        if val_acc > best_acc:
            best_acc = val_acc
            print('-- Saving model --')
            torch.save(model.state_dict(), params.save_path / f'{params.model_name}.pth')
        print('-- Saving model history --')
        with open(params.save_path / f'{params.model_name}_model_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print(f'Elapsed epoch time: {passed_time(time.time() - epoch_start_time)}')
        print(f'Total elapsed time: {passed_time(time.time() - start_time)}')
    test_acc, test_loss = eval_model(model, data_loader['test'], loss_fn, device)
    print(f'test accuracy: {test_acc:.2f}, validation loss: {test_loss:.2f}')
    history['test_acc'].append(test_acc)
    history['test_loss'].append(test_loss)
    print('-- Saving model history --')
    with open(params.save_path / f'{params.model_name}_model_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print(f'Total training time: {passed_time(time.time() - start_time)}')


def train_epoch(model, data_loader, device, optimizer, loss_fn, lr_scheduler, num_labels):
    """

    :param model:
    :param data_loader:
    :param device:
    :param optimizer:
    :param loss_fn:
    :param lr_scheduler:
    :param num_labels:
    :return:
    """
    model.train(mode=True)
    num_correct = torch.tensor(0.0)
    batch_losses = []

    num_samples = len(data_loader) * data_loader.batch_size * num_labels
    for batch in data_loader:
        optimizer.zero_grad()

        batch_num_correct, batch_loss = epoch_pass(batch, model, loss_fn, device)
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        num_correct += batch_num_correct
        batch_losses.append(batch_loss.item())

    accuracy = num_correct.item() / num_samples
    mean_loss = np.mean(batch_losses)
    return accuracy, mean_loss


def eval_model(model, data_loader, loss_fn, device, num_labels):
    """

    :param model:
    :param data_loader:
    :param loss_fn:
    :param device:
    :param num_labels:
    :return:
    """
    model.eval()
    num_correct = torch.tensor(0.0)
    batch_losses = []

    num_samples = len(data_loader) * data_loader.batch_size * num_labels

    with torch.no_grad():
        for batch in data_loader:
            batch_num_correct, batch_loss = epoch_pass(batch, model, loss_fn, device)
            batch_losses.append(batch_loss.item())
            num_correct += batch_num_correct
    accuracy = num_correct.double() / num_samples
    mean_loss = np.mean(batch_losses)
    return accuracy, mean_loss


def epoch_pass(batch, model, loss_fn, device):
    """
    Returns number of correct predictions and loss
    :param batch:
    :param model:
    :param loss_fn:
    :param device:
    :return:
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["encoded_genres"].to(device)

    batch_probs = model(input_ids, attention_mask)
    predictions = binary_labeling(batch_probs, threshold=0.5)
    return torch.sum(torch.eq(predictions, targets)), loss_fn(predictions, targets)


def binary_labeling(p, threshold):
    """

    :param p:
    :param threshold:
    :return:
    """
    res = p.clone()
    res[p >= threshold] = 1
    res[p < threshold] = 0
    return res


def passed_time(t):
    """

    :param t:
    :return:
    """
    if t < 60:
        return f'{t} seconds'
    if t // 60 < 60:
        return f'{t // 60} minutes and {passed_time(t % 60)}'
    return f'{t // 3600} hours {passed_time(t % 3600)}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training a model for genre labeling",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', help='Path to data directory', type=Path
                        , default=Path.cwd() / 'data')
    parser.add_argument('-fn', '--filename',
                        help='Filename containing data (filename ending is also required): example:'
                             ' data.csv', type=str, default='prepared_movie_data.csv')
    parser.add_argument('-mfn', '--mapping_filename',
                        help='Filename containing mapping of genres to indices (filename ending is also required): example:'
                             ' data_mapping.json', type=str, default='genre_mapping.json')
    parser.add_argument('-sp', '--save_path', help='Path to directory in which to save/load results to/from', type=Path
                        , default=Path.cwd() / 'trained_models')
    parser.add_argument('-lm', '--load_model',
                        help='Continue training from trained model', action='store_true')
    parser.add_argument('-mn', '--model_name',
                        help='Name to save/load model,'
                             ' NOTE: .pth ending is added', type=str, default='genre_classifier')
    parser.add_argument('-ptmn', '--pre_trained_model_name',
                        help='name of pre_trained_name', type=str, default='bert-base-cased')
    parser.add_argument('-drp', '--dropout',
                        help='Dropout', type=float, default=0.3)
    parser.add_argument('-menln', '--max_encoding_length',
                        help='Max length of encoding tensor -- note 512 is the max allowed number ', type=int, default=512)
    parser.add_argument('-bsz', '--batch_size',
                        help='Batch size', type=int, default=8)
    parser.add_argument('-ne', '--n_epochs',
                        help='Number of training epochs', type=int, default=10)
    parser.add_argument('-trs', '--train_split',
                        help='Ratio of training / test split', type=float, default=0.8)
    parser.add_argument('-tes', '--test_split',
                        help='Ratio of validation / test split', type=float, default=0.5)
    parser.add_argument('-rs', '--random_state',
                        help='Random state, seed', type=int, default=42)

    args = parser.parse_args()
    if not args.save_path.exists():
        args.save_path.mkdir(parents=True)
    print('-- Entered Arguments --')
    for arg in vars(args):
        print(f'- {arg}: {getattr(args, arg)}')
    parameters = ModelParameters(pre_trained_model_name=args.pre_trained_model_name,
                                 max_encoding_length=max(512, args.max_encoding_length), dropout=args.dropout,
                                 batch_size=args.batch_size, n_epochs=args.n_epochs, train_split=args.train_split,
                                 test_split=args.test_split, random_state=args.random_state, save_path=args.save_path,
                                 model_name=args.model_name)
    train_model(path=args.path, filename=args.filename, mapping_filename=args.mapping_filename, params=parameters,
                load_model=args.load_model)
