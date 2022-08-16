from models.base import BaseModel


class TextCnnModel(BaseModel):
    def __init__(self, config, word_vectors, vocab_size, output_size):
        super(TextCnnModel, self).__init__(config, word_vectors, vocab_size, output_size)
