import os
import torch
from data import TrainData, EvalData
from utils.metrics import accuracy
from utils.split_data import train_test_split
from models import (LstmModel, LstmAttenModel, RCNNModel, CharCNNModel,
                    TextCNNModel, TransformerModel, FastTextModel)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Train:
    def __init__(self, config) -> None:
        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        self.config = config
        self.data_path = self.config.DATA_PATH
        self._output_path = os.path.join(BASE_DIR, self.config.OUTPUT_PATH)
        self.train_file = os.path.join(self._output_path, 'train_data.pkl')
        self.eval_file = os.path.join(self._output_path, 'eval_data.pkl')

        self.load_data()
        self._vocab_size = self.train_data_obj.vocab_size
        self._output_size = self.config.OUTPUT_SIZE
        self.train_data, self.eval_data = train_test_split(self.data_path)
        self.train_text, self.train_labels = self.train_data_obj.generate_data(self.train_data, self.train_file,
                                                                               train=True)
        self.eval_text, self.eval_labels = self.eval_data_obj.generate_data(self.eval_data, self.eval_file)
        self.get_model()

    def load_data(self):
        self.train_data_obj = TrainData(self.config)
        self.eval_data_obj = EvalData(self.config)

    def get_model(self):
        if not self._vocab_size:
            word_to_index, label_to_index = self.train_data_obj.load_vocab()
            self._vocab_size = len(word_to_index)
            self._output_size = len(label_to_index)

        if self.config.MODEL.lower() == 'lstm':
            self.model = LstmModel(self.config, self._vocab_size, self._output_size)
        elif self.config.MODEL.lower() == 'lstm_atten':
            self.model = LstmAttenModel(self.config, self._vocab_size, self._output_size)
        elif self.config.MODEL.lower() == 'rcnn':
            self.model = RCNNModel(self.config, self._vocab_size, self._output_size)
        elif self.config.MODEL.lower() == 'charcnn':
            self.model = CharCNNModel(self.config, self._vocab_size, self._output_size)
        elif self.config.MODEL.lower() == 'textcnn':
            self.model = TextCNNModel(self.config, self._vocab_size, self._output_size)
        elif self.config.MODEL.lower() == 'transformer':
            self.model = TransformerModel(self.config, self._vocab_size, self._output_size)
        elif self.config.MODEL.lower() == 'fasttext':
            self.model = FastTextModel(self.config, self._vocab_size, self._output_size)

    def get_optimizer(self):
        if self.config.OPTIMIZER == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE)
        return None

    def train(self):
        optimizer = self.get_optimizer()
        for epoch in range(self.config.EPOCHS):
            print(f'EPOCH: {epoch + 1}/{self.config.EPOCHS}')
            i = 0
            for batch in self.train_data_obj.get_batch(self.train_text, self.train_labels):
                optimizer.zero_grad()
                predictions = self.model(batch['x'])
                predictions = torch.max(predictions, 1)[1]
                acc = accuracy(predictions, batch['y'])
                loss = self.model.calculate_loss(predictions, batch['y'])
                loss.backward()
                if i == 0:
                    print(acc)
                    print(predictions)
                    print(batch['y'])
                i += 1
