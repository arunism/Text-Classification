from models.base import BaseModel

class BiLstmModel(BaseModel):
    def __init__(self, constant, word_vectors):
        super(BiLstmModel, self).__init__(constant, word_vectors)