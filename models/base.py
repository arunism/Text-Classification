import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, config, word_vectors):
        self.config = config
        self.word_vectors = word_vectors
        self.input_size = self.config.SEQUENCE_LEN
        self.vocab_size = self.config.VOCAB_SIZE
        self.embedding_size = config.EMBED_SIZE
        self.loss = 0.0
        # self.embedding = nn.Embedding(self.input_size, self.embedding_size)
    
    def get_optimizer(self):
        if self.config.OPTIMIZER == 'adam':
            return torch.optim.Adam(lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'adadelta':
            return torch.optim.Adadelta(lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'adagrad':
            return torch.optim.Adagrad(lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'rmsprop':
            return torch.optim.RMSprop(lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'sgd':
            return torch.optim.SGD(lr=self.config.LEARNING_RATE)
        return None
    
    def calculate_loss(self, prediction, target):
        loss = nn.CrossEntropyLoss()
        output = loss(prediction, target)
        return output
