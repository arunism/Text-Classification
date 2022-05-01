import os
from base import EvalDataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EvalData(EvalDataBase):
    def __init__(self, config) -> None:
        super(EvalData, self).__init__(config)
        self._eval_data_path = os.path.join(BASE_DIR, config['eval_data_path'])
        self._output_path = os.path.join(BASE_DIR, config['output_path'])
        self._stop_word_path = os.path.join(BASE_DIR, config['stop_word_path']) if config['stop_word_path'] else None
        self._sequence_length = config['sequence_length']