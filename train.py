import os
import constants
from utils.split_data import train_test_split
from data import TrainData, EvalData
from models import BiLstmModel, BiLstmAttenModel, RCnnModel, TextCnnModel, TransformerModel

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

        self.load_data()

    def load_data(self):
        self.train_data_obj = TrainData(self.constant)
        self.eval_data_obj = EvalData(self.constant)
        self.train_data, self.eval_data = train_test_split(self.data_path)
        # text, label = self.train_data_obj.read_data(self.train_data)
        # self.train_data_obj.clean_text(text)
    
    def get_model(self):
        if self.constant.MODEL == 'bilstm':
            self.model = BiLstmModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'bilstm_atten':
            self.model = BiLstmAttenModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'rcnn':
            self.model = RCnnModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'textcnn':
            self.model = TextCnnModel(constant=self.constant, word_vectors=self.word_vectors)
        elif self.constant.MODEL == 'transformer':
            self.model = TransformerModel(constant=self.constant, word_vectors=self.word_vectors)

    def train(self):
        pass


if __name__ == '__main__':
    trainer = Train()