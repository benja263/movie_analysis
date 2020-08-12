from torch import nn
from transformers import BertModel


class GenreClassifier(nn.Module):
    def __init__(self, params):
        super(GenreClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(params.pre_trained_model_name)
        self.drop = nn.Dropout(p=params.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, params.num_labels)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)




