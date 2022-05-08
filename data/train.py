import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from data.base import DataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TrainData(DataBase):
    def __init__(self, constant) -> None:
        super(TrainData, self).__init__(constant)
    
    def get_word_vectors(self, vocab):
        word_vectors = (1/np.sqrt(self._vocab_size)*(2*np.random.rand(self._vocab_size, self._embedding_size) - 1))
    
    def build_vocab(self, words, labels):
        if os.path.exists(os.path.join(self.constant.OUTPUT_PATH, 'word_vectors.npy')):
            self.word_vectors = np.load(os.path.join(self._output_path, "word_vectors.npy"))
        
        if os.path.exists(os.path.join(self.constant.OUTPUT_PATH, 'word_to_index.pkl')) and \
           os.path.exists(os.path.join(self.constant.OUTPUT_PATH, 'label_to_index.pkl')):
            with open(os.path.join(self.constant.OUTPUT_PATH, 'word_to_index.pkl'), 'rb') as file:
                word_to_index = pickle.load(file)
            
            with open(os.path.join(self.constant.OUTPUT_PATH, 'label_to_index.pkl'), 'rb') as file:
                label_to_index = pickle.load(file)

            return word_to_index, label_to_index
        
        vocab = ['<PAD>', '<UNK>'] + words
        if not self._vocab_size: self._vocab_size = len(vocab)