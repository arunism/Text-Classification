from models.base import BaseModel

class RCnnModel(BaseModel):
    def __init__(self, constant, word_vectors):
        super(RCnnModel, self).__init__(constant, word_vectors)