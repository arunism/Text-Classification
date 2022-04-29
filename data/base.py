class Dataset:
    def __init__(self, config) -> None:
        self.config = config
    
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
    def __init__(self, config) -> None:
        super(TrainDataBase, self).__init__(config)
    
    def remove_stop_words(self, inputs):
        raise NotImplementedError
    
    def get_word_vectors(self, words):
        raise NotImplementedError
    
    def get_vocab(self, words, labels):
        raise NotImplementedError


class EvalDataBase(Dataset):
    def __init__(self, config) -> None:
        super(EvalDataBase, self).__init__(config)
    
    def load_vocab(self):
        raise NotImplementedError