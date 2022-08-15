from models.base import BaseModel


class LstmAttenModel(BaseModel):
    def __init__(self, config, word_vectors):
        super(LstmAttenModel, self).__init__(config, word_vectors)
