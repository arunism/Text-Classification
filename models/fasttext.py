from models.base import BaseModel


class FastTextModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(FastTextModel, self).__init__(config, vocab_size, output_size)
