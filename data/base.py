import os
import re
import pickle
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataBase:
    def __init__(self, constant) -> None:
        self._output_path = os.path.join(BASE_DIR, constant.OUTPUT_PATH)
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)
        self._stop_word_path = os.path.join(BASE_DIR, constant.STOPWORD_PATH) if constant.STOPWORD_PATH else None
        self._word_vec_path = os.path.join(BASE_DIR, constant.WORD_VEC_PATH)
        self._sequence_length = constant.SEQUENCE_LEN
        self._vocab_size = constant.VOCAB_SIZE
        self._embedding_size = constant.EMBED_SIZE
        self._batch_size = constant.BATCH_SIZE
        self._min_word_count = constant.MIN_WORD_COUNT
        self._text_header = constant.TEXT_HEADER
        self._label_header = constant.LABEL_HEADER
    
    def read_data(self, data):
        text = data[self._text_header].map(str)
        labels = data[self._label_header].map(str)
        return text.tolist(), labels.tolist()
    
    @staticmethod
    def clean_punct(self, sentence):
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)
        sentence = re.sub(r'[?|$|.|!]',r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]',r' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return sentence
    
    def clean_text(self, text):
        words = [word for data in text for word in self.clean_punct(data.lower()).split()]
        word_count = Counter(words)
        sorted_words = sorted(word_count.items(), key=lambda x:x[1], reverse=True)
        words = [item[0] for item in sorted_words if item[1] >= self._min_word_count]

        if self._stop_word_path:
            with open(self._stop_word_path, 'r', encoding='utf-8') as sw:
                stopwords = [line.strip() for line in sw.readlines()]
            words = [word for word in words if word not in stopwords]
        return words

    @staticmethod
    def all_text_to_index(text, word_to_index):
        idx = [
            [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
            for sentence in text
        ]
        return idx
    
    @staticmethod
    def all_label_to_index(labels, label_to_index):
        idx = [label_to_index[label] for label in labels]
        return idx
    
    def padding(self, text):
        text = [
            sentence[:self._sequence_length] if len(sentence) > self._sequence_length
            else sentence + [0]*(self._sequence_length - len(sentence))
            for sentence in text
        ]
        return text

    def load_vocab(self):
        w2i_file = os.path.join(self._output_path, 'word_to_index.pkl')
        l2i_file = os.path.join(self._output_path, 'label_to_index.pkl')
        with open(w2i_file, 'rb') as file: word_to_index = pickle.load(file)
        with open(l2i_file, 'rb') as file: label_to_index = pickle.load(file)
        return word_to_index, label_to_index
    
    def get_batch(self, x, y):
        a = np.arange(len(x))
        np.random.shuffle(a)
        x, y = x[a], y[a]
        num_of_batches = len(x) // self._batch_size
        for i in range(num_of_batches):
            start = i*self._batch_size
            end = start + self._batch_size
            batch_x = np.array(x[start:end], dtype='int64')
            batch_y = np.array(y[start:end], dtype='float32')
            yield dict(x=batch_x, y=batch_y)
