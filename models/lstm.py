from models.base import BaseModel


class LstmModel(BaseModel):
    def __init__(self, config, word_vectors):
        super(LstmModel, self).__init__(config, word_vectors)
