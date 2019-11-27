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
from keras.layers import SeparableConv1D
from keras.layers import MaxPooling1D
from keras.initializers import Constant
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.preprocessing import text


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
            'max_sequence_length': The maximum length of all sequences. 
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
    max_sequence_length = len(max(sequences, key=len))
    if max_sequence_length > MAX_SEQ_LENGTH:
        max_sequence_length = MAX_SEQ_LENGTH
    
    # Pad the sequences to the maximum length
    sequences = sequence.pad_sequences(sequences, maxlen=max_sequence_length)
    
    # Create an dictionary to hold additional information
    info = {}
    info['tokenizer'] = tokenizer
    info['vocab_size'] = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    info['max_sequence_length'] = max_sequence_length

    return (sequences, labels), info


def load_embedding(embedding_file, language, word_index, vocab_size):
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

        print("Found {} fastText word vectors.".format(len(embedding)))

        # Build the embedding matrix of the passed vocab
        embedding_dim = len(next(embedding.values()))
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        oov_count = 0
        for word, i in word_index.items():
            if i >= vocab_size:
                continue
            vector = embedding.get(word)
            if vector is not None:
                embedding_matrix[i] = vector
            else:
                # Words not found in the embedding will be assigned to 0 vectors
                embedding_matrix[i] = np.zeros(300)
                oov_count += 1

        print ('Embedding out of vocabulary words: {}'.format(oov_count))

        return embedding_matrix


def sep_cnn_model(input_shape,
                  num_classes,
                  num_features,
                  embedding_matrix,
                  blocks=1,
                  filters=64,
                  kernel_size=4,
                  dropout_rate=0.5):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    model = models.Sequential()
    model.add(Embedding(input_dim=num_features, output_dim=300, input_length=input_shape,
                        embeddings_initializer=Constant(embedding_matrix)))

    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=3))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))

    model.add(GlobalAveragePooling1D())
    # model.add(MaxPooling1D())
    model.add(Dropout(rate=0.5))
    model.add(Dense(op_units, activation=op_activation))
    return model

# One-hot encoding to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


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
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        # Model
        self.initialized_model = False
        self.model = None

        # Training data
        self.train_sequences = None
        self.train_labels = None

        # Testing data
        self.test_sequences = None
        self.test_labels = None

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
       
        if not self.initialized_model:
            # Preprocess data
            (self.train_sequences, self.train_labels), info = preprocess_training_data(train_dataset, metadata['language'])
            vocab_size = info['vocab_size']

            # Load pretrained embedding
            embedding_file = ''
            if metadata['language'] == 'EN':
                embedding_file = 'cc.en.300.vec.gz'
            else:
                embedding_file = 'cc.zh.300.vec.gz'
            embedding_matrix = load_embedding(embedding_file, metadata['language'], info['tokenizer'].word_index, vocab_size)

            # Initialize model
            model = sep_cnn_model(input_shape=x_train.shape[1:][0],
                                  num_classes=num_classes,
                                  num_features=num_features,
                                  embedding_matrix=embedding_matrix,
                                  blocks=2,
                                  filters=64,
                                  kernel_size=4,
                                  dropout_rate=0.5)
            if num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
            optimizer = tf.keras.optimizers.Adam(lr=1e-3)
            model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10)]

            self.initialized_model = True

        x_train, y_train = shuffle(x_train, y_train)
        # fit model
        history = model.fit(
            x_train,
            ohe2cat(y_train),
            # y_train,
            epochs=1000,
            callbacks=callbacks,
            validation_split=0.2,
            # validation_data=(x_dev,y_dev),
            verbose=2,  # Logs once per epoch.
            batch_size=32,
            shuffle=True)
        print(str(type(x_train)) + " " + str(y_train.shape))

        # save model
        model.save(self.train_output_path + 'model.h5')
        with open(self.train_output_path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.train_output_path + 'model.config', 'wb') as f:
            f.write(str(max_length).encode())
            f.close()

        self.done_training = True

    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        model = models.load_model(self.test_input_path + 'model.h5')
        with open(self.test_input_path + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle, encoding='iso-8859-1')
        with open(self.test_input_path + 'model.config', 'r') as f:
            max_length = int(f.read().strip())
            f.close()

        train_num, test_num = self.metadata['train_num'], self.metadata['test_num']
        class_num = self.metadata['class_num']

        # tokenizing Chinese words
        if self.metadata['language'] == 'ZH':
            x_test = clean_zh_text(x_test)
            x_test = list(map(_tokenize_chinese_words, x_test))
        else:
            x_test = clean_en_text(x_test)

        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=max_length)
        result = model.predict_classes(x_test)

        # category class list to sparse class list of lists
        y_test = np.zeros([test_num, class_num])
        for idx, y in enumerate(result):
            y_test[idx][y] = 1
        return y_test

