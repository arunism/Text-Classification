from models.base import BaseModel


class TextCnnModel(BaseModel):
    def __init__(self, config, word_vectors):
        super(TextCnnModel, self).__init__(config, word_vectors)
