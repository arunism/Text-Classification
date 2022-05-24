from models.base import BaseModel

class LstmModel(BaseModel):
    def __init__(self, constant, word_vectors):
        super(LstmModel, self).__init__(constant, word_vectors)