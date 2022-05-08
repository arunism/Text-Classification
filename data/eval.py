import os
from data.base import DataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EvalData(DataBase):
    def __init__(self, constant) -> None:
        super(EvalData, self).__init__(constant)
    
    def load_vocab(self):
        raise NotImplementedError