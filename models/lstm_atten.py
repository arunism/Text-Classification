from models.base import BaseModel


class LstmAttenModel(BaseModel):
    def __init__(self, constant, word_vectors):
        super(LstmAttenModel, self).__init__(constant, word_vectors)
