import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config, word_vectors, vocab_size, output_size):
        super(BaseModel, self).__init__()
        self.config = config
        self.word_vectors = word_vectors
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_size = self.config.EMBED_SIZE
        self.loss = 0.0
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # Define Layers
        self.fc1 = nn.Linear(self.embedding_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.output_size)
        self.init_weights(0.5)

    def init_weights(self, init_range):
        self.get_weight_initializer(self.embedding, init_range, is_embedding=True)
        self.get_weight_initializer(self.fc1, init_range)
        self.get_weight_initializer(self.fc2, init_range)
        self.get_weight_initializer(self.fc3, init_range)

    def get_weight_initializer(self, layer, init_range, is_embedding=False):
        if self.config.WEIGHT_INITIALIZATION == 'uniform':
            layer.weight.data.uniform_(-init_range, init_range)
        else:
            layer.weight.data.normal_(-init_range, init_range)
        if not is_embedding:
            layer.bias.data.zero_()
    
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
