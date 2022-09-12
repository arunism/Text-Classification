# You can choose between ['lstm', 'lstm_atten', 'rcnn', 'charcnn', 'textcnn', 'transformer', 'fasttext']
MODEL = 'lstm'
# You can choose between ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']
OPTIMIZER = 'adam'
# You can choose between ['uniform', 'normal'] distributions
WEIGHT_INITIALIZATION = 'uniform'
EPOCHS = 10
LEARNING_RATE = 0.001
SEQUENCE_LEN = 350
EMBED_SIZE = 200
HIDDEN_SIZE = 256
BATCH_SIZE = 128
DROPOUT = 0.4
NUM_LAYERS = 2
VOCAB_SIZE = None
OUTPUT_SIZE = None
DATA_PATH = 'dataset/data.csv'
STOPWORD_PATH = 'stopwords/english.txt'
OUTPUT_PATH = 'results/'
TEXT_HEADER = 'Summary'
LABEL_HEADER = 'Genre'
TRAIN_TEST_SPLIT_RATIO = 0.8
MIN_WORD_COUNT = 2
