import torch

from utils.utils import A_minus_intersection, cosine_similarity, load_model, load_json


class MovieClassifier:
    """

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
        encoding = self.metadata['tokenizer'].encode_plus(plot_summary, add_special_tokens=True,
                                                          max_length=self.metadata['parameters']['max_encoding_length'],
                                                          return_token_type_ids=False, pad_to_max_length=True,
                                                          return_attention_mask=True, return_tensors='pt',
                                                          truncation=True)
        input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
        probabilities = torch.sigmoid(self.model(input_ids, attention_mask)).flatten()
        probability_mapping = {i: probabilities[i].item() for i in range(len(probabilities))}
        return {self.reverse_mapping[k]: v for k, v in sorted(probability_mapping.items(), key=lambda item: item[1],
                                                              reverse=True)}

    def calculate_similarities(self, movie_name):
        embeddings = self.embeddings
        if movie_name not in self.similarity_cache.keys() and movie_name in embeddings.keys():
            similarity_dict = dict()
            movie_embedding = embeddings.get(movie_name, None)
            movie_names = A_minus_intersection(set(embeddings.keys()), set([movie_name]))
            for other_movie_name in movie_names:
                other_embedding = embeddings[other_movie_name]
                similarity_dict[other_movie_name] = cosine_similarity(movie_embedding, other_embedding, is_tensor=False)
            self.similarity_cache[movie_name] = {k: v for k, v in sorted(similarity_dict.items(), key=lambda item: item[1],
                                                 reverse=True)}

    def get_n_most_similar(self, movie_name, N):
        if movie_name not in self.similarity_cache.keys():
            self.calculate_similarities(movie_name)
        similarities = self.similarity_cache[movie_name]
        return {k: similarities[k] for k in list(similarities.keys())[:N]}


from pathlib import Path

path = Path('/Users/benjaminfuhrer/GitHub/movie_analysis/trained_models')
data_path = Path('/Users/benjaminfuhrer/GitHub/movie_analysis/data')
filename = 'genre_classifier'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, metadata = load_model(path, filename, device)
embeddings = load_json(data_path, 'embeddings.json')

plot_summary = "When bratty 8-year-old Kevin McCallister (Macaulay Culkin) acts out the night before a family trip" \
               " to Paris, his mother (Catherine O'Hara) makes him sleep in the attic. After the McCallisters mistakenly " \
               "leave for the airport without Kevin, he awakens to an empty house and assumes his wish to have no" \
               " family has come true. But his excitement sours when he realizes that two con men" \
               " (Joe Pesci, Daniel Stern) plan to rob the McCallister residence, and that he alone must protect the" \
               " family home."
movie_classifier = MovieClassifier(model, metadata, device, embeddings)
prediction = movie_classifier.predict_genre_by_plot(plot_summary)
print(prediction)
print(movie_classifier.get_n_most_similar('Home Alone', 20))
