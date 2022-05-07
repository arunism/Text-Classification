from models.base import BaseModel

class TextCnnModel(BaseModel):
    def __init__(self, constant, word_vectors):
        super(TextCnnModel, self).__init__(constant, word_vectors)