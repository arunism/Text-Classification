import os
import constants
from data.train import TrainData
from data.eval import EvalData
from utils.split_data import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Train:
    def __init__(self) -> None:
        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        self.train_data = None
        self.eval_data = None
        self.constant = constants
        self.data_path = self.constant.DATA_PATH

        self.load_data()

    def load_data(self):
        self.train_data_obj = TrainData(self.constant)
        self.eval_data_obj = EvalData(self.constant)
        self.train_data, self.eval_data = train_test_split(self.data_path)

if __name__ == '__main__':
    trainer = Train()