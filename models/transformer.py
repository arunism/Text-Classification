from models.base import BaseModel


class TransformerModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(TransformerModel, self).__init__(config, vocab_size, output_size)
