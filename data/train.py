import os
import pickle
import numpy as np
from data.base import DataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TrainData(DataBase):
    def __init__(self, constant) -> None:
        super(TrainData, self).__init__(constant)
        self.word_vectors = None
        self.train_file = os.path.join(self._output_path, 'train_data.pkl')
        self.w2i_file = os.path.join(self._output_path, 'word_to_index.pkl')
        self.l2i_file = os.path.join(self._output_path, 'label_to_index.pkl')
    
    def get_word_vectors(self, vocab):
        pass

    def _build_vocab(self, words, labels):
        if os.path.exists(self.w2i_file) and os.path.exists(self.l2i_file):
            word_to_index, label_to_index = self.load_vocab()
            return word_to_index, label_to_index
        
        vocab = ['<PAD>', '<UNK>'] + words
        if not self._vocab_size:
            self._vocab_size = len(vocab)

        word_to_index = dict(zip(vocab, list(range(self._vocab_size))))
        label_to_index = dict(zip(list(set(labels)), list(range(len(list(set(labels)))))))
        with open(self.w2i_file, 'wb') as file: pickle.dump(word_to_index, file)
        with open(self.l2i_file, 'wb') as file: pickle.dump(label_to_index, file)
        return word_to_index, label_to_index
    
    def generate_data(self, data):
        if os.path.exists(self.train_file) and os.path.exists(self.w2i_file) and os.path.exists(self.l2i_file):
            with open(self.train_file, 'rb') as file: train_data = pickle.load(file)
            word_to_index, label_to_index = self.load_vocab()
            return np.array(train_data['text_idx']), np.array(train_data['label_idx']), label_to_index
        
        text, labels = self.read_data(data)
        words = self.clean_text(text)
        text = [self.clean_punct(sentence) for sentence in text]
        word_to_index, label_to_index = self._build_vocab(words, labels)
        text_idx = self.all_text_to_index(text, word_to_index)
        text_idx = self.padding(text_idx)
        label_idx = self.all_label_to_index(labels, label_to_index)
        train_data = dict(text_idx=text_idx, label_idx=label_idx)
        with open(self.train_file, 'wb') as file: pickle.dump(train_data, file)
        return np.array(text_idx), np.array(label_idx), label_to_index
