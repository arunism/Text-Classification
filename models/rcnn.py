from models.base import BaseModel


class RCNNModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(RCNNModel, self).__init__(config, vocab_size, output_size)
