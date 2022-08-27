from models.base import BaseModel


class TextCNNModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(TextCNNModel, self).__init__(config, vocab_size, output_size)
