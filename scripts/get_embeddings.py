import argparse
import json
from pathlib import Path

import torch

from utils.helpers import A_minus_intersection, load_json, save_json, load_model


def get_embeddings(model_path, data_path, model_filename):
    print('-- Device -- ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print('-- Loading Model --')
    model, metadata = load_model(model_path, model_filename, device)
    print('-- Loading Plot Summaries --')
    with open(data_path / 'plot_summaries.json', 'rb') as f:
        plot_summaries = json.load(f)
    print('-- Getting Embeddings --')
    if not plot_summaries:
        print("-- Can't get embeddings -> No movies found --")
        return
    embeddings = load_json(model_path, 'embeddings.json')
    print(f'Length of embeddings: {len(embeddings.keys())}')
    missing_movie_names = A_minus_intersection(set(plot_summaries.keys()), set(embeddings.keys()))
    plot_summaries = {k: v for k, v in plot_summaries.items() if k in missing_movie_names}
    print(f'Number of movie names to get embeddings for: {len(plot_summaries.keys())}')
    model.eval()
    with torch.no_grad():
        for ind, (movie_name, plot_summary) in enumerate(plot_summaries.items(), start=1):
            if movie_name not in embeddings.keys():
                encoding = metadata['tokenizer'].encode_plus(plot_summary, add_special_tokens=True,
                                                             max_length=metadata['parameters'][
                                                                 'max_encoding_length'],
                                                             return_token_type_ids=False, pad_to_max_length=True,
                                                             return_attention_mask=True, return_tensors='pt',
                                                             truncation=True)
                input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
                embedding = model.extract_embedding(input_ids, attention_mask).flatten()
                embeddings[movie_name] = embedding.tolist()
            if ind % 1000 == 0 or ind == len(plot_summaries.keys()):
                # batch update
                print(f'Saving batch {ind}/{len(plot_summaries.keys())}')
                print(f'Length of embeddings: {len(embeddings.keys())}')
                save_json(embeddings, model_path, 'embeddings.json')
    print(f'Length of embeddings: {len(embeddings.keys())}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for getting embeddings of movie plots",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp', '--model_path', help='Path to model directory', type=Path
                        , default=Path.cwd() / 'trained_models')
    parser.add_argument('-dp', '--data_path', help='Path to data', type=Path
                        , default=Path.cwd() / 'data')
    parser.add_argument('-fn', '--filename',
                        help='model filename', type=str, default='genre_classifier')
    args = parser.parse_args()

    get_embeddings(model_path=args.model_path, data_path=args.data_path, model_filename=args.filename)
