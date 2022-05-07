from asyncio import constants
from models.base import BaseModel

class BiLstmAttenModel(BaseModel):
    def __init__(self, constant, word_vectors):
        super(BiLstmAttenModel, self).__init__(constant, word_vectors)