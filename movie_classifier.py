"""
Module for predicting movie genres calculating similarities of a given movie
"""
import torch

from utils.helpers import A_minus_intersection
from utils.vector_similarities import cosine_similarity, euclidean_distance, dot_product

SIMILARITY_FUNC = {'cosine': cosine_similarity, 'distance': euclidean_distance, 'dot': dot_product}


class MovieClassifier:
    """
    Class for predicting movie genres by plot summaries and finding the N-most similar movies
    """
    def __init__(self, model, model_metadata, device, embeddings):
        self.model = model
        self.metadata = model_metadata
        self.movie_similarity_mapping = None
        self.device = device
        self.reverse_mapping = {v: k for k, v in model_metadata['genre_mapping'].items()}
        self.embeddings = embeddings
        self.similarity_cache = dict()

    def predict_genre_by_plot(self, plot_summary):
        """
        Return predicted genres for a given plot summary (genres are sorted and returned in a descending order)
        :param str plot_summary:
        :return:
        """
        encoding = self.metadata['tokenizer'].encode_plus(plot_summary, add_special_tokens=True,
                                                          max_length=self.metadata['parameters']['max_encoding_length'],
                                                          return_token_type_ids=False, pad_to_max_length=True,
                                                          return_attention_mask=True, return_tensors='pt',
                                                          truncation=True)
        input_ids, attention_mask = encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
        probabilities = torch.sigmoid(self.model(input_ids, attention_mask)).flatten()
        probability_mapping = {i: probabilities[i].item() for i in range(len(probabilities))}
        return {self.reverse_mapping[k]: v for k, v in sorted(probability_mapping.items(), key=lambda item: item[1],
                                                              reverse=True)}

    def calculate_similarities(self, movie_name, similarity_type):
        """
        Calculates similarity scores
        :param str movie_name:
        :param str similarity_type: similarity score method ['cosine', 'dot', 'distance']
        :return:
        """
        embeddings = self.embeddings
        if movie_name not in self.similarity_cache.keys() and movie_name in embeddings.keys():
            similarity_dict = dict()
            movie_embedding = embeddings.get(movie_name, None)
            movie_names = A_minus_intersection(set(embeddings.keys()), {movie_name})
            for other_movie_name in movie_names:
                other_embedding = embeddings[other_movie_name]
                similarity_dict[other_movie_name] = SIMILARITY_FUNC[similarity_type](movie_embedding,
                                                                                     other_embedding, is_tensor=False)
            self.similarity_cache[movie_name] = {k: v for k, v in sorted(similarity_dict.items(), key=lambda item: item[1],
                                                 reverse=True if similarity_type != 'distance' else False)}

    def get_n_most_similar(self, movie_name, N, similarity_type='cosine'):
        """
        Return N-most similar movies for a given movie name
        :param str movie_name:
        :param int N: number of top similar movies to return
        :param str similarity_type: similarity score method ['cosine', 'dot', 'distance']
        :return:
        """
        if movie_name not in self.similarity_cache.keys():
            self.calculate_similarities(movie_name, similarity_type)
        similarities = self.similarity_cache[movie_name]
        return {k: similarities[k] for k in list(similarities.keys())[:N]}
