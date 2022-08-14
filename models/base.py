import torch
import torch.nn as nn


class BaseModel:
    def __init__(self, constant, word_vectors):
        self.constant = constant
        self.word_vectors = word_vectors
        self.vocab_size = self.constant.VOCAB_SIZE
        self._embedding_size = constant.EMBED_SIZE
        self.loss = 0.0
    
    def get_optimizer(self):
        if self.constant.OPTIMIZER == 'adam':
            return torch.optim.Adam(lr=self.constant.LEARNING_RATE)
        elif self.constant.OPTIMIZER == 'adadelta':
            return torch.optim.Adadelta(lr=self.constant.LEARNING_RATE)
        elif self.constant.OPTIMIZER == 'adagrad':
            return torch.optim.Adagrad(lr=self.constant.LEARNING_RATE)
        elif self.constant.OPTIMIZER == 'rmsprop':
            return torch.optim.RMSprop(lr=self.constant.LEARNING_RATE)
        elif self.constant.OPTIMIZER == 'sgd':
            return torch.optim.SGD(lr=self.constant.LEARNING_RATE)
        return None
    
    def calculate_loss(self, prediction, target):
        loss = nn.CrossEntropyLoss()
        output = loss(prediction, target)
        return output
