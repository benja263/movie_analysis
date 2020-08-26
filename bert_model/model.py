"""
Module containing a neural net model structure based on pretrained BERT with an added logit output layer for
 multi-label classification
"""
from torch import nn
from transformers import BertModel


class MultiGenreLabeler(nn.Module):
    def __init__(self, params):
        super(MultiGenreLabeler, self).__init__()
        self.bert = BertModel.from_pretrained(params.pre_trained_model_name)
        self.drop = nn.Dropout(p=params.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, params.num_labels)

    def forward(self, input_ids, attention_mask):
        """
        :param torch.tensor input_ids:
        :param torch.tensor attention_mask: 1 for text-token, 0 for padded-token
        :return:
        """
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

    def extract_embedding(self, input_ids, attention_mask):
        """
        Returns trained embeddings represented by the first token of last hidden layer of BERT
        :param torch.tensor input_ids:
        :param torch.tensor attention_mask: 1 for text-token, 0 for padded-token
        :return:
        """
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return pooled_output




