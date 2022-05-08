import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from data.base import DataBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TrainData(DataBase):
    def __init__(self, constant) -> None:
        super(TrainData, self).__init__(constant)
        self._word_vectors = None
    
    def get_word_vectors(self, vocab):
        word_vectors = (1/np.sqrt(self._vocab_size)*(2*np.random.rand(self._vocab_size, self._embedding_size) - 1))
        if os.path.splitext(self._word_vec_path)[-1] == '.bin':
            vectors = KeyedVectors.load_word2vec_format(self._word_vec_path, binary=True)
        else:
            vectors = KeyedVectors.load_word2vec_format(self._word_vec_path)
        for i in range(self._vocab_size):
            vector = vectors.wv[vocab[i]]
            word_vectors[i, :] = vector
            return word_vectors

    def _build_vocab(self, words, labels):
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

        if self._word_vec_path:
            self._word_vectors = self.get_word_vectors(vocab)
            np.save(os.path.join(self._output_path, 'word_vectors.npy'), self._word_vectors)

        word_to_index = dict(zip(vocab, list(range(self._vocab_size))))
        label_to_index = dict(zip(list(set(labels)), list(range(len(list(set(labels)))))))
        with open(os.path.join(self._output_path, 'word_to_index.pkl'), 'wb') as file: pickle.dump(word_to_index, file)
        with open(os.path.join(self._output_path, 'label_to_index.pkl'), 'wb') as file: pickle.dump(label_to_index, file)
        return word_to_index, label_to_index
    
    def generate_train_data(self, data):
        train_file = os.path.join(self._output_path, 'train_data.pkl')
        w2i_file = os.path.join(self._output_path, 'word_to_index.pkl')
        l2i_file = os.path.join(self._output_path, 'label_to_index.pkl')
        word_vec_file = os.path.join(self._output_path, 'word_vectors.npy')

        if os.path.exists(train_file) and os.path.exists(w2i_file) and os.path.exists(l2i_file):
            with open(train_file, 'rb') as file: train_data = pickle.load(file)
            with open(w2i_file, 'rb') as file: word_to_index = pickle.load(file)
            with open(l2i_file, 'rb') as file: label_to_index = pickle.load(file)
            if os.path.exists(word_vec_file): self._word_vectors = np.load(word_vec_file)
            return np.array(train_data['text_idx']), np.array(train_data['label_idx']), label_to_index
        
        text, labels = self.read_data(data)
        words = self.clean_text(text)
        word_to_index, label_to_index = self._build_vocab(words, labels)
        text_idx = self.all_text_to_index(text, word_to_index)
        text_idx = self.padding(text_idx, self._sequence_length)
        label_idx = self.all_label_to_index(labels, label_to_index)
        train_data = dict(text_idx, label_idx)
        with open(train_file, 'wb') as file: pickle.dump(train_data, file)
        return np.array(text_idx), np.array(label_idx), label_to_index