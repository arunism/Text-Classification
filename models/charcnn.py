from models.base import BaseModel


class CharCNNModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(CharCNNModel, self).__init__(config, vocab_size, output_size)
