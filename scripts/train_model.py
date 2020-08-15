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
    # debugging = True
    # if debugging:
    #     data = data.sample(frac=0.001, replace=False)
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
                                                  params.batch_size, plot_col='plot_summary', genre_col='genres',
                                                  num_workers=params.num_workers)
    val_data_loader = create_genres_data_loader(val_data, mapping, tokenizer, params.max_encoding_length,
                                                params.batch_size, plot_col='plot_summary', genre_col='genres',
                                                num_workers=params.num_workers)
    test_data_loader = create_genres_data_loader(test_data, mapping, tokenizer, params.max_encoding_length,
                                                 params.batch_size, plot_col='plot_summary', genre_col='genres',
                                                 num_workers=params.num_workers)
    return {'train': train_data_loader, 'validation': val_data_loader, 'test': test_data_loader}


def training(model, data_loader, params, num_labels, load_model, device):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=params.learning_rate, correct_bias=False)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                   num_training_steps=len(data_loader['train']) * params.n_epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    model_info = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'device': device,
                  'loss_fn': loss_fn}
    best_f1 = 0.0
    history = defaultdict(list)
    history['best_f1'].append(best_f1)
    start_epoch = 1
    if load_model:
        print('-- Continuing model training --')
        with open(params.save_path / f'{params.model_name}_model_history.pkl', 'rb') as f:
            history = pickle.load(f)
            start_epoch = history['epoch'][-1] + 1
            best_f1 = history['best_f1'][-1]
            print(f'Best f1 score: {best_f1}')

    start_time = time.time()
    for i in range(start_epoch, start_epoch + params.n_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {i}/{start_epoch + params.n_epochs - 1}')
        print('-' * 10)
        tr_prec, tr_recall, tr_acc, tr_f1, tr_loss = train_epoch(**model_info, data_loader=data_loader['train'],
                                                                 num_labels=num_labels)
        val_prec, val_recall, val_acc, val_f1, val_loss = eval_model(model, data_loader['validation'],
                                                                     loss_fn, device, num_labels)

        history['epoch'].append(i)
        history['tr_acc'].append(tr_acc)
        history['tr_prec'].append(tr_prec)
        history['tr_recall'].append(tr_recall)
        history['tr_f1'].append(tr_f1)
        history['tr_loss'].append(tr_loss)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)
        print('-- Training metrics -- ')
        print(
            f'accuracy: {tr_acc:.2f}, precision: {tr_prec:.2f}, recall: {tr_recall:.2f}, f1: {tr_f1:.2f} training loss: {tr_loss:.2f}')
        print('-- Validation metrics -- ')
        print(
            f'accuracy: {val_acc:.2f}, precision: {val_prec:.2f}, recall: {val_recall:.2f}, f1: {val_f1:.2f} training loss: {val_loss:.2f}')
        print('-' * 10)
        if val_f1 > best_f1:
            best_f1 = val_f1
            history['best_f1'][-1] = best_f1
            print('-- Saving model --')
            torch.save(model.state_dict(), params.save_path / f'{params.model_name}.pth')
        print('-- Saving model history --')
        with open(params.save_path / f'{params.model_name}_model_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print(f'Elapsed epoch time: {passed_time(time.time() - epoch_start_time)}')
        print(f'Total elapsed time: {passed_time(time.time() - start_time)}')
        print(f'Best f1 score: {best_f1}')
    test_prec, test_recall, test_acc, test_f1, test_loss = eval_model(model, data_loader['test'], loss_fn, device)
    print('-- Test metrics-')
    print(
        f'accuracy: {test_acc:.2f}, precision: {test_prec:.2f}, recall: {test_recall:.2f}, f1: {test_f1:.2f} training loss: {test_loss:.2f}')
    history['test_acc'].append(test_acc)
    history['test_prec'].append(test_prec)
    history['test_recall'].append(test_recall)
    history['test_f1'].append(test_f1)
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
    batch_losses = []
    batch_precs, batch_recalls, batch_accs, batch_f1s = [], [], [], []
    for batch in data_loader:
        optimizer.zero_grad()
        conf_matrix, batch_loss = epoch_pass(batch, model, loss_fn, device)

        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        tp, tn, fp, fn = conf_matrix
        precision, recall, accuracy, f1 = metrics(tp.item(), tn.item(), fp.item(), fn.item())
        batch_precs.append(precision)
        batch_recalls.append(recall)
        batch_accs.append(accuracy)
        batch_f1s.append(f1)
        batch_losses.append(batch_loss.item())

    mean_loss = np.mean(batch_losses)
    mean_precision, mean_recall, mean_accuray = np.mean(batch_precs), np.mean(batch_recalls), np.mean(batch_accs)
    mean_f1 = np.mean(batch_f1s)
    return mean_precision, mean_recall, mean_accuray, mean_f1, mean_loss


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
    batch_losses = []
    batch_precs, batch_recalls, batch_accs, batch_f1s = [], [], [], []

    with torch.no_grad():
        for batch in data_loader:
            conf_matrix, batch_loss = epoch_pass(batch, model, loss_fn, device)
            tp, tn, fp, fn = conf_matrix
            precision, recall, accuracy, f1 = metrics(tp.item(), tn.item(), fp.item(), fn.item())
            batch_precs.append(precision)
            batch_recalls.append(recall)
            batch_accs.append(accuracy)
            batch_f1s.append(f1)
            batch_losses.append(batch_loss.item())
    mean_loss = np.mean(batch_losses)
    mean_precision, mean_recall, mean_accuray = np.mean(batch_precs), np.mean(batch_recalls), np.mean(batch_accs)
    mean_f1 = np.mean(batch_f1s)
    return mean_precision, mean_recall, mean_accuray, mean_f1, mean_loss


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

    batch_logits = model(input_ids, attention_mask)
    predictions = binary_labeling(batch_logits, threshold=0.5, device=device)
    return confusion_matrix(predictions, targets), loss_fn(batch_logits, targets)


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


def confusion_matrix(predictions, targets):
    """

    :param predictions:
    :param targets:
    :return:
    """
    TP = torch.sum((predictions == 1) & (targets == 1), dtype=torch.float)
    TN = torch.sum((predictions == 0) & (targets == 0), dtype=torch.float)
    FP = torch.sum((predictions == 1) & (targets == 0), dtype=torch.float)
    FN = torch.sum((predictions == 0) & (targets == 1), dtype=torch.float)
    return TP, TN, FP, FN


def metrics(TP, TN, FP, FN):
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
    return precision, recall, accuracy, f1


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
                        help='Max length of encoding tensor -- note 512 is the max allowed number ', type=int,
                        default=512)
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
    parser.add_argument('-lr', '--learning_rate',
                        help='Optimizer learning rate', type=float, default=2e-5)
    parser.add_argument('-nm', '--num_workers',
                        help='How many subprocesses to use for data loading', type=int, default=4)

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
                                 model_name=args.model_name, learning_rate=args.learning_rate,
                                 num_workers=args.num_workers)
    train_model(path=args.path, filename=args.filename, mapping_filename=args.mapping_filename, params=parameters,
                load_model=args.load_model)
