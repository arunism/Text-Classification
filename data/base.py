import os
import re
import torch
import pickle
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataBase:
    def __init__(self, config) -> None:
        self._output_path = os.path.join(BASE_DIR, config.OUTPUT_PATH)
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)
        self._stop_word_path = os.path.join(BASE_DIR, config.STOPWORD_PATH) if config.STOPWORD_PATH else None
        self._sequence_length = config.SEQUENCE_LEN
        self.vocab_size = config.VOCAB_SIZE
        self._batch_size = config.BATCH_SIZE
        self._min_word_count = config.MIN_WORD_COUNT
        self._text_header = config.TEXT_HEADER
        self._label_header = config.LABEL_HEADER
        self.w2i_file = os.path.join(self._output_path, 'word_to_index.pkl')
        self.l2i_file = os.path.join(self._output_path, 'label_to_index.pkl')
    
    def read_data(self, data):
        text = data[self._text_header].map(str)
        labels = data[self._label_header].map(str)
        return text.tolist(), labels.tolist()

    def clean_punct(self, sentence):
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)
        sentence = re.sub(r'[?|$|.|!]', r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return sentence
    
    def clean_text(self, text):
        words = [word for data in text for word in self.clean_punct(data.lower()).split()]
        word_count = Counter(words)
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sorted_words if item[1] >= self._min_word_count]

        if self._stop_word_path:
            with open(self._stop_word_path, 'r', encoding='utf-8') as sw:
                stopwords = [line.strip() for line in sw.readlines()]
            words = [word for word in words if word not in stopwords]
        return words

    def padding(self, text):
        text = [
            sentence[:self._sequence_length] if len(sentence) > self._sequence_length
            else sentence + [0]*(self._sequence_length - len(sentence))
            for sentence in text
        ]
        return text

    def all_text_to_index(self, text, word_to_index):
        idx = [
            [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
            for sentence in text
        ]
        idx = self.padding(idx)
        return np.array(idx, dtype='int64')
    
    @staticmethod
    def all_label_to_index(labels, label_to_index):
        idx = [label_to_index[label] for label in labels]
        return np.array(idx, dtype='float32')

    def load_vocab(self):
        w2i_file = os.path.join(self._output_path, 'word_to_index.pkl')
        l2i_file = os.path.join(self._output_path, 'label_to_index.pkl')
        with open(w2i_file, 'rb') as file: word_to_index = pickle.load(file)
        with open(l2i_file, 'rb') as file: label_to_index = pickle.load(file)
        return word_to_index, label_to_index

    def _build_vocab(self, words, labels):
        if os.path.exists(self.w2i_file) and os.path.exists(self.l2i_file):
            word_to_index, label_to_index = self.load_vocab()
            return word_to_index, label_to_index

        vocab = ['<PAD>', '<UNK>'] + words
        if not self.vocab_size:
            self.vocab_size = len(vocab)

        word_to_index = dict(zip(vocab, list(range(self.vocab_size))))
        label_to_index = dict(zip(list(set(labels)), list(range(len(list(set(labels)))))))
        with open(self.w2i_file, 'wb') as file:
            pickle.dump(word_to_index, file)
        with open(self.l2i_file, 'wb') as file:
            pickle.dump(label_to_index, file)
        return word_to_index, label_to_index

    def generate_data(self, data, file, train=False):
        if os.path.exists(file):
            with open(file, 'rb') as file: file_data = pickle.load(file)
            return file_data['text_idx'], file_data['label_idx']

        text, labels = self.read_data(data)
        text = [self.clean_punct(sentence) for sentence in text]
        if train:
            words = self.clean_text(text)
            word_to_index, label_to_index = self._build_vocab(words, labels)
        else:
            word_to_index, label_to_index = self.load_vocab()
        text_idx = self.all_text_to_index(text, word_to_index)
        label_idx = self.all_label_to_index(labels, label_to_index)
        data = dict(text_idx=text_idx, label_idx=label_idx)
        with open(file, 'wb') as file: pickle.dump(data, file)
        return text_idx, label_idx
    
    def get_batch(self, x, y):
        a = np.arange(len(x))
        np.random.shuffle(a)
        x, y = x[a], y[a]
        num_of_batches = len(x) // self._batch_size
        for i in range(num_of_batches):
            start = i*self._batch_size
            end = start + self._batch_size
            batch_x = torch.from_numpy(x[start:end])
            batch_y = torch.from_numpy(y[start:end])
            yield dict(x=batch_x, y=batch_y)
