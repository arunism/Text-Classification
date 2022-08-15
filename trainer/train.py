import os
import config
from utils.split_data import train_test_split
from data import TrainData, EvalData
from models import LstmModel, LstmAttenModel, RCnnModel, TextCnnModel, TransformerModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Train:
    def __init__(self) -> None:
        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        self.word_vectors = None
        self.config = config
        self.data_path = self.config.DATA_PATH
        self._output_path = os.path.join(BASE_DIR, self.config.OUTPUT_PATH)
        self.train_file = os.path.join(self._output_path, 'train_data.pkl')
        self.eval_file = os.path.join(self._output_path, 'eval_data.pkl')

        self.load_data()
        self.train_data, self.eval_data = train_test_split(self.data_path)
        self.train_text, self.train_labels = self.train_data_obj.generate_data(self.train_data, self.train_file,
                                                                               train=True)
        self.eval_text, self.eval_labels = self.eval_data_obj.generate_data(self.eval_data, self.eval_file)
        self._vocab_size = self.config.VOCAB_SIZE
        self.get_model()

    def load_data(self):
        self.train_data_obj = TrainData(self.config)
        self.eval_data_obj = EvalData(self.config)

    def get_model(self):
        if self.config.MODEL == 'lstm':
            self.model = LstmModel(config=self.config, word_vectors=self.train_text)
        elif self.config.MODEL == 'lstm_atten':
            self.model = LstmAttenModel(config=self.config, word_vectors=self.train_text)
        elif self.config.MODEL == 'rcnn':
            self.model = RCnnModel(config=self.config, word_vectors=self.train_text)
        elif self.config.MODEL == 'textcnn':
            self.model = TextCnnModel(config=self.config, word_vectors=self.train_text)
        elif self.config.MODEL == 'transformer':
            self.model = TransformerModel(config=self.config, word_vectors=self.train_text)

    def train(self):
        for epoch in range(self.config.EPOCHS):
            print(f'EPOCH: {epoch + 1}/{self.config.EPOCHS}')
            for batch in self.train_data_obj.get_batch(self.train_text, self.train_labels):
                pass
