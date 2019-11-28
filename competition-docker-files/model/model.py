import pandas as pd
import os
import gzip
import argparse
import time
import re
import jieba
import pickle
import tensorflow as tf
import numpy as np
import sys, getopt
from subprocess import check_output
from keras import models
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.optimizers import Adagrad
from keras.initializers import Constant
from keras.preprocessing import text
from keras.preprocessing import sequence


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

EMBEDDINGS_DIR = "/app/embedding"
MAX_SEQ_LENGTH = 500
MAX_VOCAB_SIZE = 20000 # Limit on the number of features. We use the top 20K features
NUM_EPOCHS_PER_TRAIN = 1
BATCH_SIZE = 32


# Functions to clean examples both in English and Chinese. Heavily inspired on the Baseline 2.
# Code from https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
def clean_en_examples(examples):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    tokenization_clean = lambda ex: ' '.join(jieba.cut(ex, cut_all=False))
    
    cleaned = []
    for ex in examples:
        ex = ex.lower()
        ex = REPLACE_BY_SPACE_RE.sub(' ', ex)
        ex = BAD_SYMBOLS_RE.sub('', ex)
        ex = ex.strip()
        cleaned.append(ex)
    return cleaned


def clean_zh_examples(examples):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    tokenization_clean = lambda ex: ' '.join(jieba.cut(ex, cut_all=False))
    
    cleaned = []
    for ex in examples:
        ex = REPLACE_BY_SPACE_RE.sub(' ', ex)
        ex = ex.strip()
        cleaned.append(tokenization_clean(ex))
    return cleaned


def preprocess_training_data(raw_data, language):
    """Preprocesses training data in both English and Chinese.

    This functions first cleans the text data, then fits a tokenizer on the cleaned examples.

    Args:
        raw_data: A tuple of (examples, labels) on which the model will train.
        language: The language of the text. 'EN' for English and 'ZN' for Chinese.

    Returns:
        A tuple of (sequences, labels) in which sequences are the preprocessed examples.
        A dictionary containing additional information on the preprocessing. It contains the following keys:
            'tokenizer': The preprocessing.text.Tokenizer object used to fit the examples.
            'vocab_size': The size of the vocabulary fitted on the examples.
            'max_seq_length': The maximum length of all sequences. 
    """
    examples, labels = raw_data

    # Clean examples' text
    if language == 'EN':
        examples = clean_en_examples(examples)
    else:
        examples = clean_zh_examples(examples)

    # Create a tokenizer on the examples corpus
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(examples)
    sequences = tokenizer.texts_to_sequences(examples)

    # Get the maximum length on these sequences
    max_seq_length = len(max(sequences, key=len))
    if max_seq_length > MAX_SEQ_LENGTH:
        max_seq_length = MAX_SEQ_LENGTH
    
    # Pad the sequences to the maximum length
    sequences = sequence.pad_sequences(sequences, maxlen=max_seq_length)
    
    # Convert one hot encoding labels to categorical labels
    labels = np.argmax(labels, axis=1)

    # Create an dictionary to hold additional information
    info = {}
    info['tokenizer'] = tokenizer
    info['vocab_size'] = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    info['max_seq_length'] = max_seq_length

    return (sequences, labels), info


def load_embedding(embedding_file, language):
    # Load pretrained embedding
    embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_file)

    # Read file and construct lookup table
    with gzip.open(embedding_path, 'rb') as f:
        embedding = {}

        for line in f.readlines():
            values = line.strip().split()
            if language == 'ZH':
                word = values[0].decode('utf8')
            else:
                word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding[word] = vector

        print("Found {} fastText word vectors for language {}.".format(len(embedding), language))
        return embedding


def emb_mlp_model(vocab_size,
                  input_length,
                  num_classes,
                  embedding_matrix,
                  hidden_layer_units,
                  dropout_rate=0.5):

    embedding_dim = embedding_matrix.shape[1]
    
    # Instantiate model and embedding layer
    model = models.Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length,
                        embeddings_initializer=Constant(embedding_matrix)))
    
    # Average the embeddings of all words per example
    model.add(GlobalAveragePooling1D())

    # Add the hidden layers
    for num_units in hidden_layer_units:
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(num_units, activation='relu'))

    # Add the final layer
    last_units, last_activation = None, None
    if num_classes == 2:
        last_units, last_activation = 1, 'sigmoid'
    else:
        last_units, last_activation = num_classes, 'softmax'
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(last_units, activation=last_activation))

    return model


class Model(object):
    """ 
        model of CNN baseline without pretraining.
        see `https://aclweb.org/anthology/D14-1181` for more information.
    """

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "train_num": 10000,
             "test_num": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        # Added attributes
        self.input_info = None
        self.model = None
        self.callbacks = []
        self.train_x = None
        self.train_y = None

        # Load embeddings
        self.embedding = None
        if metadata['language'] == 'EN':
            self.embedding = load_embedding('cc.en.300.vec.gz', 'EN')
        else:
            self.embedding = load_embedding('cc.zh.300.vec.gz', 'ZH')

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.
        
        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        # If the model was not initialized
        if self.model is None:
            # Preprocess data
            (self.train_x, self.train_y), self.input_info = preprocess_training_data(train_dataset, self.metadata['language'])
            vocab_size = self.input_info['vocab_size']
            input_length = self.input_info['max_seq_length']
            num_classes = self.metadata['class_num']

            # Build the embedding matrix of the passed vocab
            word_index = self.input_info['tokenizer'].word_index
            embedding_dim = len(next(iter(self.embedding.values())))
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            oov_count = 0
            for word, i in word_index.items():
                if i >= vocab_size:
                    continue
                vector = self.embedding.get(word)
                if vector is not None:
                    embedding_matrix[i] = vector
                else:
                    # Words not found in the embedding will be assigned to vectors of zeros
                    embedding_matrix[i] = np.zeros(300)
                    oov_count += 1
            print('Embedding out of vocabulary words: {}'.format(oov_count))

            # Initialize model
            model = emb_mlp_model(vocab_size,
                                       input_length,
                                       num_classes,
                                       embedding_matrix,
                                       hidden_layer_units=[1000])

            # Define optimizer and compile model
            if num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
            optimizer = Adagrad()
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            
            self.model = model
        else:
            # Load model
            model = models.load_model(self.test_input_path + 'model.h5')
            with open(self.test_input_path + 'tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle, encoding='iso-8859-1')

        # Train model
        history = model.fit(
            x=self.train_x,
            y=self.train_y,
            epochs=NUM_EPOCHS_PER_TRAIN,
            callbacks=self.callbacks,
            validation_split=0.2,
            verbose=2,  # Logs once per epoch.
            batch_size=BATCH_SIZE,
            shuffle=True)
            
        # Save model
        model.save(self.train_output_path + 'model.h5')
        with open(self.train_output_path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.input_info['tokenizer'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self, test_x, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        # Load model
        model = models.load_model(self.test_input_path + 'model.h5')
        with open(self.test_input_path + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle, encoding='iso-8859-1')
        num_test, num_classes = self.metadata['test_num'], self.metadata['class_num']
        max_seq_length = self.input_info['max_seq_length']

        # Clean examples' text
        if self.metadata['language'] == 'EN':
            test_x = clean_en_examples(test_x)
        else:
            test_x = clean_zh_examples(test_x)

        # Tokenize and pad examples
        test_x = tokenizer.texts_to_sequences(test_x)
        test_x = sequence.pad_sequences(test_x, maxlen=max_seq_length)

        # Evaluate model
        result = model.predict_classes(test_x)

        # Convert to one hot encoding
        y_test = np.zeros((num_test, num_classes))
        for idx, y in enumerate(result):
            y_test[idx][y] = 1
        return y_test
