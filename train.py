import os
import yaml
from data.train import TrainData
from data.eval import EvalData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Train:
    def __init__(self) -> None:
        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        self.config = yaml.safe_load(open(os.path.join(BASE_DIR, 'config.yml')))

    def load_data(self):
        self.train_data_obj = TrainData(self.config)
        self.eval_data_obj = EvalData(self.config)

if __name__ == '__main__':
    trainer = Train()