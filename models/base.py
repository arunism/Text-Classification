import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, config, vocab_size, output_size):
        super(BaseModel, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.batch_size = self.config.BATCH_SIZE
        self.input_size = self.config.SEQUENCE_LEN
        self.hidden_size = self.config.HIDDEN_SIZE
        self.num_layers = self.config.NUM_LAYERS
        self.embedding_size = self.config.EMBED_SIZE
        self.loss = 0.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
    
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
