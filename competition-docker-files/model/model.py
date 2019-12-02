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
config.log_device_placement = False # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

EMBEDDINGS_DIR = "/app/embedding"
MAX_SEQ_LENGTH = 500
MAX_VOCAB_SIZE = 20000 # Limit on the number of features. We use the top 20K features
NUM_EPOCHS_PER_TRAIN = 2
BATCH_SIZE = 32


# Functions to clean instances both in English and Chinese. Heavily inspired on the Baseline 2.
# Code from https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
def clean_en_instances(instances):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    tokenization_clean = lambda ex: ' '.join(jieba.cut(ex, cut_all=False))
    
    cleaned = []
    for instance in instances:
        instance = instance.lower()
        instance = REPLACE_BY_SPACE_RE.sub(' ', instance)
        instance = BAD_SYMBOLS_RE.sub('', instance)
        instance = instance.strip()
        cleaned.append(instance)
    return cleaned


def clean_zh_instances(instances):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    tokenization_clean = lambda instance: ' '.join(jieba.cut(instance, cut_all=False))
    
    cleaned = []
    for instance in instances:
        instance = REPLACE_BY_SPACE_RE.sub(' ', instance)
        instance = instance.strip()
        cleaned.append(tokenization_clean(instance))
    return cleaned


def get_tokenizer(instances, language):
    # Clean text
    if language == 'EN':
        instances = clean_en_instances(instances)
    else:
        instances = clean_zh_instances(instances)

    # Create a tokenizer on the instances corpus
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(instances)
    sequences = tokenizer.texts_to_sequences(instances)

    # Get the maximum length on these sequences
    max_seq_length = len(max(sequences, key=len))
    if max_seq_length > MAX_SEQ_LENGTH:
        max_seq_length = MAX_SEQ_LENGTH

    # Get the vocab size
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    
    return tokenizer, vocab_size, max_seq_length


def preprocess_text(instances, tokenizer, max_seq_length, language):
    # Clean text
    if language == 'EN':
        instances = clean_en_instances(instances)
    else:
        instances = clean_zh_instances(instances)

    # Apply tokenizer to text
    sequences = tokenizer.texts_to_sequences(instances)

    # Pad sequences
    sequences = sequence.pad_sequences(sequences, maxlen=max_seq_length)

    return sequences


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


def get_emb_mlp_model(vocab_size,
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
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    optimizer = Adagrad()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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
        self.max_seq_length = None
        self.tokenizer = None
        self.model = None
        self.x_train = None
        self.x_test = None

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
                     here `sample_count` is the number of instances in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        if self.model is None:
            # If the model was not initialized
            num_classes = self.metadata['class_num']
            x_train, y_train = train_dataset

            # Get the tokenizer based on the training instances
            self.tokenizer, vocab_size, self.max_seq_length = get_tokenizer(x_train, self.metadata['language']) 

            # Build the embedding matrix of the vocab
            word_index = self.tokenizer.word_index
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
            self.model = get_emb_mlp_model(vocab_size,
                                           self.max_seq_length,
                                           num_classes,
                                           embedding_matrix,
                                           hidden_layer_units=[1000])

        if self.x_train is None:
            self.x_train = preprocess_text(x_train, self.tokenizer, self.max_seq_length, self.metadata['language'])
        
        # Train model
        history = self.model.fit(
            x=self.x_train,
            y=y_train,
            epochs=NUM_EPOCHS_PER_TRAIN,
            validation_split=0.2,
            verbose=2,  # Logs once per epoch.
            batch_size=BATCH_SIZE,
            shuffle=True)
            
    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of instances in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        if self.x_test is None:
            self.x_test = preprocess_text(x_test, self.tokenizer, self.max_seq_length, self.metadata['langauge'])

        # Evaluate model
        return model.predict(self.x_test)
