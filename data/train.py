import os
from base import TrainDataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TrainData(TrainDataBase):
    def __init__(self, config) -> None:
        super(TrainData, self).__init__(config)
        self._train_data_path = os.path.join(BASE_DIR, config['train_data_path'])
        self._output_path = os.path.join(BASE_DIR, config['output_path'])
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)
        self._word_vector_path = os.path.join(BASE_DIR, config['word_vector_path']) if config['word_vector_path'] else None
        self._stop_word_path = os.path.join(BASE_DIR, config['stop_word_path']) if config['stop_word_path'] else None
        self._sequence_length = config['sequence_length']
        self._vocab_size = config['vocab_size']
        self._embedding_size = config['embedding_size']
        self._batch_size = config['batch_size']
        self.word_vector = None