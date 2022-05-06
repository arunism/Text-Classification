import os
from data.base import TrainDataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TrainData(TrainDataBase):
    def __init__(self, constant) -> None:
        super(TrainData, self).__init__(constant)