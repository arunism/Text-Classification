from models.base import BaseModel


class LstmAttenModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(LstmAttenModel, self).__init__(config, vocab_size, output_size)
