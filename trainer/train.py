import os
import constants
from utils.split_data import train_test_split
from data import TrainData, EvalData
from models import LstmModel, LstmAttenModel, RCnnModel, TextCnnModel, TransformerModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Train:
    def __init__(self) -> None:
        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        self.train_data = None
        self.eval_data = None
        self.word_vectors = None
        self.constant = constants
        self.data_path = self.constant.DATA_PATH
        self._output_path = os.path.join(BASE_DIR, self.constant.OUTPUT_PATH)
        self.train_file = os.path.join(self._output_path, 'train_data.pkl')
        self.eval_file = os.path.join(self._output_path, 'eval_data.pkl')

        self.load_data()
        self.train_text, self.train_labels = self.train_data_obj.generate_data(self.train_data, self.train_file,
                                                                               train=True)
        self.eval_text, self.eval_labels = self.eval_data_obj.generate_data(self.eval_data, self.eval_file)
        self._vocab_size = self.constant.VOCAB_SIZE
        self.get_model()

    def load_data(self):
        self.train_data_obj = TrainData(self.constant)
        self.eval_data_obj = EvalData(self.constant)
        self.train_data, self.eval_data = train_test_split(self.data_path)

    def get_model(self):
        if self.constant.MODEL == 'lstm':
            self.model = LstmModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'lstm_atten':
            self.model = LstmAttenModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'rcnn':
            self.model = RCnnModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'textcnn':
            self.model = TextCnnModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'transformer':
            self.model = TransformerModel(constant=self.constant, word_vectors=self.word_vectors)

    def train(self):
        for epoch in range(self.constant.EPOCHS):
            print(f'EPOCH: {epoch + 1}/{self.constant.EPOCHS}')
            for batch in self.train_data_obj.get_batch(self.train_text, self.train_labels):
                pass