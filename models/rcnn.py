from models.base import BaseModel


class RCnnModel(BaseModel):
    def __init__(self, config, word_vectors, vocab_size, output_size):
        super(RCnnModel, self).__init__(config, word_vectors, vocab_size, output_size)
