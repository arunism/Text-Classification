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
        inputs = self.dropout(self.embedding(word_vectors))
        # Converting from shape (batch_size, seq_len,  embed_size) to (seq_len, batch_size,  embed_size)
        inputs = inputs.permute(1, 0, 2)
        # Initializing hidden and cell state for lstm
        h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))
        c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))
        output, (hidden, cell) = self.lstm(inputs, (h0, c0))
        output = self.output(hidden[-1])
        return output
