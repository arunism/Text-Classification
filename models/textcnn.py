from models.base import BaseModel


class TextCnnModel(BaseModel):
    def __init__(self, config, vocab_size, output_size):
        super(TextCnnModel, self).__init__(config, vocab_size, output_size)
