from models.base import BaseModel


class LstmAttenModel(BaseModel):
    def __init__(self, config, word_vectors, vocab_size):
        super(LstmAttenModel, self).__init__(config, word_vectors, vocab_size)
