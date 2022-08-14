import os
from data.base import DataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TrainData(DataBase):
    def __init__(self, constant) -> None:
        super(TrainData, self).__init__(constant)
