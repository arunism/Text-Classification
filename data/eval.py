import os
from data.base import EvalDataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EvalData(EvalDataBase):
    def __init__(self, constant) -> None:
        super(EvalData, self).__init__(constant)