import os
import re
from collections import Counter
from tracemalloc import stop
from constants import TEXT_HEADER, LABEL_HEADER

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataBase:
    def __init__(self, constant) -> None:
        self._output_path = os.path.join(BASE_DIR, constant.OUTPUT_PATH)
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)
        self._stop_word_path = os.path.join(BASE_DIR, constant.STOPWORD_PATH) if constant.STOPWORD_PATH else None
        self._sequence_length = constant.SEQUENCE_LEN
        self._vocab_size = constant.VOCAB_SIZE
        self._embedding_size = constant.EMBED_SIZE
        self._batch_size = constant.BATCH_SIZE
        self._min_word_count = constant.MIN_WORD_COUNT
    
    def read_data(self, data):
        text = data[TEXT_HEADER].map(str)
        labels = data[LABEL_HEADER].map(str)
        return text.tolist(), labels.tolist()

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
    def trans_to_index(text, word_to_index):
        raise NotImplementedError
    
    @staticmethod
    def trans_label_to_index(text, label_to_index):
        raise NotImplementedError
    
    def padding(self, text, seq_len):
        raise NotImplementedError
    
    def generate_data(self):
        raise NotImplementedError
    
    def next_batch(self, x, y, batch_size):
        raise NotImplementedError