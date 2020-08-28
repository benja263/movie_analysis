"""
Module for training a BERT model
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import attr
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

from bert_model.genres_dataset import create_genres_data_loader
from bert_model.model import MultiGenreLabeler
from bert_model.parameters import ModelParameters
from utils.helpers import passed_time, save_data
from utils.serialization import load_json, save_pickle, save_model, load_model
from utils.metrics import confusion_matrix, classification_metrics


def train_model(path, output_dir, filename, params, continue_training):
    """

    :param Path path: data path
    :param str filename: model name
    :param ModelParameters params:
    :param bool continue_training: continue training a model or start from scratch
    :return:
    """
    print('-- Loading Data --')
    data = pd.read_csv(path / 'prepared_movie_data.csv')
    # debugging = True
    # if debugging:
    #     data = data.sample(frac=0.0005, replace=False)
    #     params.num_workers = 0
    mapping = load_json(path, 'genre_mapping.json')
    print('-- Splitting Data --')
    print(f'train data ratio: {params.train_split},'
          f' validation_ratio: {((1 - params.train_split) * params.validation_split):.2f}, '
          f'test_ratio: {((1.0 - params.train_split) * (1.0 - params.validation_split)):.2f} ')
    train_data, test_data = train_test_split(data, train_size=params.train_split, shuffle=True,
                                             random_state=params.random_state)
    val_data, test_data = train_test_split(test_data, train_size=params.validation_split, shuffle=True,
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
    print('-- Training Model -- ')
    params.num_labels = len(mapping)

    metadata = {'tokenizer': tokenizer,
                'genre_mapping': mapping,
                'model_type': 'torch',
                'parameters':  attr.asdict(params)}
    training(filename, output_dir, data_loaders, params, continue_training, metadata)


def get_data_loaders(data, mapping, tokenizer, params):
    """
    Return Training/Validation/Test data loaders
    :param tuple(pd.DataFrame) data: train, validation and test data
    :param dict mapping: genre to index mapping
    :param BertTokenizer tokenizer: pre-trained bert tokenizer
    :param ModelParameters params:
    :return:
    """
    train_data, val_data, test_data = data
    train_data_loader = create_genres_data_loader(train_data, mapping, tokenizer, params.max_encoding_length,
                                                  params.batch_size, plot_col='plot_summary', genre_col='movie_genres',
                                                  num_workers=params.num_workers)
    val_data_loader = create_genres_data_loader(val_data, mapping, tokenizer, params.max_encoding_length,
                                                params.batch_size, plot_col='plot_summary', genre_col='movie_genres',
                                                num_workers=params.num_workers)
    test_data_loader = create_genres_data_loader(test_data, mapping, tokenizer, params.max_encoding_length,
                                                 params.batch_size, plot_col='plot_summary', genre_col='movie_genres',
                                                 num_workers=params.num_workers)
    return {'train': train_data_loader, 'validation': val_data_loader, 'test': test_data_loader}


def training(filename, output_dir, data_loader, params, continue_training, metadata):
    """

    :param str filename: model name
    :param Path output_dir:
    :param DataLoader data_loader: pytorch DataLoader
    :param ModelParameters params:
    :param bool continue_training: continue training a model or start from scratch
    :param dict metadata: model metadata
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')
    model, metadata = load_model(output_dir, filename, device) if continue_training else initialize_model(metadata,
                                                                                                          params,
                                                                                                          device)
    optimizer = AdamW(params=model.parameters(), lr=params.learning_rate, correct_bias=False)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                   num_training_steps=len(data_loader['train']) * params.n_epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    start_epoch = metadata['history']['epoch'][-1] + 1 if continue_training else 1
    best_f1 = max(metadata['history']['val_f1']) if continue_training else 0.0
    print(f'Continuing training: best f1 score: {best_f1:.5f}') if continue_training else print('Starting training')

    model_info = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'device': device,
                  'loss_fn': loss_fn}
    start_time = time.time()
    for i in range(start_epoch, start_epoch + params.n_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {i}/{start_epoch + params.n_epochs - 1}')
        print('-' * 10)
        tr_metrics = train_epoch(**model_info, data_loader=data_loader['train'])
        val_metrics = eval_model(model, data_loader['validation'], loss_fn, device)
        print('-- Training metrics -- ')
        print_metrics(tr_metrics)
        print('-- Validation metrics -- ')
        print_metrics(val_metrics)
        print('-' * 10)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            print('-- Saving bert_model --')
            save_model(model, output_dir, filename)
        print('-- Saving bert_model history --')
        metadata['history']['epoch'].append(i)
        metadata['history'] = append_history(metadata['history'], **tr_metrics, metric_type='tr')
        metadata['history'] = append_history(metadata['history'], **val_metrics, metric_type='val')
        save_pickle(metadata, output_dir, f'{filename}_metadata')
        print(f'Elapsed epoch time: {passed_time(time.time() - epoch_start_time)}')
        print(f'Total elapsed time: {passed_time(time.time() - start_time)}')
        print(f"Best f1 score: {max(metadata['history']['val_f1']):.5f}")
    test_metrics = eval_model(model, data_loader['test'], loss_fn, device)
    print('-- Test metrics-')
    print_metrics(test_metrics)
    metadata['history'] = append_history(metadata['history'], **test_metrics, metric_type='test')
    print('-- Saving bert_model history --')
    save_pickle(metadata, output_dir, f'{filename}_metadata')
    print(f'Total training time: {passed_time(time.time() - start_time)}')


def initialize_model(metadata, params, device):
    """
    Return model
    :param dict metadata: model metadata
    :param ModelParameters params:
    :param str device:
    :return:
    """
    model = MultiGenreLabeler(params)
    model.to(device)
    metadata['history'] = defaultdict(list)
    return model, metadata


def train_epoch(model, data_loader, device, optimizer, loss_fn, lr_scheduler):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    :param nn.Module model:
    :param DataLoader data_loader:
    :param str device:
    :param optimizer:
    :param loss_fn: loss function
    :param lr_scheduler: schedule of linear decrease in learning rate
    :return:
    """
    model.train(mode=True)
    mean_metrics = {'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'loss': 0.0}
    num_batches = len(data_loader)
    for batch in data_loader:
        optimizer.zero_grad()
        conf_matrix, batch_loss = epoch_pass(batch, model, loss_fn, device)

        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        tp, tn, fp, fn = conf_matrix
        batch_metrics = classification_metrics(tp.item(), tn.item(), fp.item(), fn.item())
        batch_metrics['loss'] = batch_loss.item()
        for metric, metric_value in batch_metrics.items():
            mean_metrics[metric] += metric_value
    for metric, metric_value in mean_metrics.items():
        mean_metrics[metric] /= num_batches
    return mean_metrics


def eval_model(model, data_loader, loss_fn, device):
    """
    Evaluate model on data and return dictionary with the average metric values
    :param nn.Module model:
    :param DataLoader data_loader:
    :param str device:
    :param loss_fn: loss function
    :param str device:
    :return:
    """
    model.eval()
    mean_metrics = {'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'loss': 0.0}
    num_batches = len(data_loader)
    with torch.no_grad():
        for batch in data_loader:
            conf_matrix, batch_loss = epoch_pass(batch, model, loss_fn, device)
            tp, tn, fp, fn = conf_matrix
            batch_metrics = classification_metrics(tp.item(), tn.item(), fp.item(), fn.item())
            batch_metrics['loss'] = batch_loss.item()
            for metric, metric_value in batch_metrics.items():
                mean_metrics[metric] += metric_value
    for metric, metric_value in mean_metrics.items():
        mean_metrics[metric] /= num_batches
    return mean_metrics


def epoch_pass(batch, model, loss_fn, device):
    """
    Returns confusion matrix and loss between the predicted output and target output
    :param DataLoader batch:
    :param nn_Module model:
    :param loss_fn: loss function
    :param str device:
    :return:
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["encoded_genres"].to(device)

    batch_logits = model(input_ids, attention_mask)
    predictions = binary_labeling(batch_logits, threshold=0.5, device=device)
    return confusion_matrix(predictions, targets, is_torch=True), loss_fn(batch_logits, targets)


def append_history(history_dict, accuracy, precision, recall, f1, loss, metric_type):
    """
    Return training history dictionary with appended metrics
    :param dict history_dict:
    :param float accuracy:
    :param float precision:
    :param float recall:
    :param float f1:
    :param float loss:
    :param str metric_type: 'tr', 'val', 'test' representing training/validation or test
    :return:
    """
    history_dict[f'{metric_type}_acc'].append(accuracy)
    history_dict[f'{metric_type}_prec'].append(precision)
    history_dict[f'{metric_type}_recall'].append(recall)
    history_dict[f'{metric_type}_f1'].append(f1)
    history_dict[f'{metric_type}_loss'].append(loss)
    return history_dict


def print_metrics(metrics):
    """
    Print values of metrics contained in dictionary of metrics
    :param dict metrics:
    :return:
    """
    for metric, metric_value in metrics.items():
        print(f"{metric}: {metric_value:.5f}", end=' ')
    print('')


def binary_labeling(logit, threshold, device):
    """
    Returns one hot encoding of class labels by evaluating the probability of each class individually
    :param torch.tensor logit:
    :param float threshold: probability threshold for assigning a given class label
    :param str device:
    :return:
    """
    res = torch.zeros(size=tuple(logit.size()), dtype=torch.float, device=device, requires_grad=False)
    res[torch.sigmoid(logit) >= threshold] = 1
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training a bert_model for genre labeling",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', help='Path to data directory', type=Path
                        , default=Path.cwd() / 'data')
    parser.add_argument('-o', '--output_dir', help='Path to directory in which to save/load results to/from', type=Path
                        , default=Path.cwd() / 'trained_models')
    parser.add_argument('-ct', '--continue_training',
                        help='Continue training from trained bert_model', action='store_true')
    parser.add_argument('-fn', '--filename',
                        help='Name to save/load bert_model,'
                             ' NOTE: .pth ending is added', type=str, default='genre_classifier')
    parser.add_argument('-ptmn', '--pre_trained_model_name',
                        help='name of pre_trained_name', type=str, default='bert-base-cased')
    parser.add_argument('-drp', '--dropout',
                        help='Dropout', type=float, default=0.3)
    parser.add_argument('-menln', '--max_encoding_length',
                        help='Max length of encoding tensor -- note 512 is the max allowed number ', type=int,
                        default=512)
    parser.add_argument('-bsz', '--batch_size',
                        help='Batch size', type=int, default=8)
    parser.add_argument('-ne', '--n_epochs',
                        help='Number of training epochs', type=int, default=10)
    parser.add_argument('-trs', '--train_split',
                        help='Ratio of training / test split', type=float, default=0.8)
    parser.add_argument('-ves', '--validation_split',
                        help='Ratio of validation / test split', type=float, default=0.5)
    parser.add_argument('-rs', '--random_state',
                        help='seed', type=int, default=42)
    parser.add_argument('-lr', '--learning_rate',
                        help='Optimizer learning rate', type=float, default=3e-5)
    parser.add_argument('-nm', '--num_workers',
                        help='How many subprocesses to use for data loading', type=int, default=4)

    args = parser.parse_args()
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    print('-- Entered Arguments --')
    for arg in vars(args):
        print(f'- {arg}: {getattr(args, arg)}')
    parameters = ModelParameters(pre_trained_model_name=args.pre_trained_model_name,
                                 max_encoding_length=max(512, args.max_encoding_length), dropout=args.dropout,
                                 batch_size=args.batch_size, n_epochs=args.n_epochs, train_split=args.train_split,
                                 validation_split=args.validation_split, random_state=args.random_state,
                                 learning_rate=args.learning_rate, num_workers=args.num_workers)
    train_model(path=args.path, filename=args.filename, params=parameters,
                continue_training=args.continue_training, output_dir=args.output_dir)

