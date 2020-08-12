import attr


@attr.s
class ModelParameters:
    pre_trained_model_name = attr.ib(default=None)
    dropout = attr.ib(default=None)
    num_labels = attr.ib(default=None)

    batch_size = attr.ib(default=None)
    max_encoding_length = attr.ib(default=None)

    n_epochs = attr.ib(default=None)
    train_split = attr.ib(default=None)
    test_split = attr.ib(default=None)
    random_state = attr.ib(default=None)

    save_path = attr.ib(default=None)

    model_name = attr.ib(default=None)




