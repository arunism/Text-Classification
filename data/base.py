import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Dataset:
    def __init__(self, constant) -> None:
        self._output_path = os.path.join(BASE_DIR, constant.OUTPUT_PATH)
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)
        self._stop_word_path = os.path.join(BASE_DIR, constant.STOPWORD_PATH) if constant.STOPWORD_PATH else None
        self._sequence_length = constant.SEQUENCE_LEN
        self._vocab_size = constant.VOCAB_SIZE
        self._embedding_size = constant.EMBED_SIZE
        self._batch_size = constant.BATCH_SIZE
    
    def read_data(self):
        raise NotImplementedError

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        raise NotImplementedError
    
    def padding(self, inputs, seq_len):
        raise NotImplementedError
    
    def generate_data(self):
        raise NotImplementedError
    
    def next_batch(self, x, y, batch_size):
        raise NotImplementedError


class TrainDataBase(Dataset):
    def __init__(self, constant) -> None:
        super(TrainDataBase, self).__init__(constant)
    
    def remove_stop_words(self, inputs):
        raise NotImplementedError
    
    def get_word_vectors(self, words):
        raise NotImplementedError
    
    def get_vocab(self, words, labels):
        raise NotImplementedError


class EvalDataBase(Dataset):
    def __init__(self, constant) -> None:
        super(EvalDataBase, self).__init__(constant)
    
    def load_vocab(self):
        raise NotImplementedError