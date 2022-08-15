from models.base import BaseModel


class RCnnModel(BaseModel):
    def __init__(self, config, word_vectors):
        super(RCnnModel, self).__init__(config, word_vectors)
