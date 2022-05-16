import os
import pickle
import numpy as np
from data.base import DataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EvalData(DataBase):
    def __init__(self, constant) -> None:
        super(EvalData, self).__init__(constant)
        self.eval_file = os.path.join(self._output_path, 'eval_data.pkl')
    
    def generate_data(self, data):
        if os.path.exists(self.eval_file):
            with open(self.eval_file, 'rb') as file: eval_data = pickle.load(file)
            return np.array(eval_data['text_idx']), np.array(eval_data['label_idx'])
        
        text, labels = self.read_data(data)
        words = self.clean_text(text)
        word_to_index, label_to_index = self.load_vocab()
        text_idx = self.all_text_to_index(text, word_to_index)
        text_idx = self.padding(text_idx, self._sequence_length)
        label_idx = self.all_label_to_index(labels, label_to_index)
        eval_data = dict(text_idx, label_idx)
        with open(self.eval_file, 'wb') as file: pickle.dump(eval_data, file)
        return np.array(text_idx), np.array(label_idx)