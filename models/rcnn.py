from models.base import BaseModel


class RCnnModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(RCnnModel, self).__init__(config, vocab_size, output_size)
