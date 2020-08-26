"""
Parameters for data preparation and training
"""
import attr


@attr.s
class ModelParameters:
    """
    Model Parameters
    - str pre_trained_model_name: name of pre_trained bert bert_model
    - float dropout: dropout ratio
    - int num_labels: number of unique possible genres
    - float learning_rate: step size at each iteration while moving toward a minimum of a loss function.
    - int max_encoding_length:  max number of tokens to encode - BERT max is 512
    """
    pre_trained_model_name = attr.ib(default=None)
    dropout = attr.ib(default=None)
    num_labels = attr.ib(default=None)
    learning_rate = attr.ib(default=None)
    max_encoding_length = attr.ib(default=None)

    """
        Training Parameters
    - int batch_size: number of samples in each batch
    - int num_workers: number of processes that generate batches in parallel
    - int n_epochs: number of training epochs
    - float train_split: ratio of training samples over training + test samples
    - float validation_split: ratio of validation samples over validation + test samples
    - int random_state: seed
    """

    batch_size = attr.ib(default=None)
    num_workers = attr.ib(default=None)
    n_epochs = attr.ib(default=None)
    train_split = attr.ib(default=None)
    validation_split = attr.ib(default=None)
    random_state = attr.ib(default=None)





