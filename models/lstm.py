import torch
from torch import nn
from torch.autograd import Variable
from models.base import BaseModel


class LstmModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(LstmModel, self).__init__(config, vocab_size, output_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, word_vectors):
        pass
