from models.base import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, constant, word_vectors):
        super(TransformerModel, self).__init__(constant, word_vectors)