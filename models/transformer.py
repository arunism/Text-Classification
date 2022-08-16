from models.base import BaseModel


class TransformerModel(BaseModel):
    def __init__(self, config, word_vectors, vocab_size):
        super(TransformerModel, self).__init__(config, word_vectors, vocab_size)
