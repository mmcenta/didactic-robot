#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import models
from tensorflow.keras import models
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence

from bert_tokenization import FullTokenizer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
tf.enable_eager_execution(config=config)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

BERT_EN_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
BERT_ZH_URL = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/1"
MAX_SEQ_LENGTH = 500
NUM_EPOCHS_PER_TRAIN = 2
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


def preprocess_examples(examples, tokenizer, language):
    """Preprocesses data in both English and Chinese.

    This functions first cleans the text data, then applies the received tokenizer on the cleaned examples.

    Args:
        raw_data: A tensor of examples.
        tokenizer: The tokenizer that will be applied to the cleaned text.
        language: The language of the text. 'EN' for English and 'ZN' for Chinese.

    Returns:
        The sequences corresponding to the tokenization of the cleaned text padded to MAX_SEQ_LENGTH.
    """
    # Clean text
    if language == 'EN':
        examples = clean_en_examples(examples)
    else:
        examples = clean_zh_examples(examples)

    # Apply tokenizer to text
    sequences = []
    for example in examples:
        sequences.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example)))

    # Pad the sequences to the maximum length
    sequences = sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)
    
    return sequences


def get_bert_classifier(num_classes, language):
    """Returns a BERT-based classifier using Keras Functional API"""

    # Create the Input objects
    input_word_ids = Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="segment_ids")
    inputs = [input_word_ids, input_mask, segment_ids]

    # Download the correct version of BERT
    if language == 'EN': 
        bert_layer = hub.KerasLayer(BERT_EN_URL, trainable=False)
    else:
        bert_layer = hub.KerasLayer(BERT_ZH_URL, trainable=False)

    # Apply the BERT layer
    _, seq_output = bert_layer(inputs) 

    # Instantiate and apply the classifier layer
    classifier_layer = Dense(num_classes, activation='softmax')
    predictions = classifier_layer(seq_output)

    model = tf.keras.Model(inputs=inputs, outputs=predictions, name='bert-classifier')

    # Compile model
    optimizer = Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Get model vocabulary and lowercase flag (for the tokenizer)
    vocab = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lowercase = bert_layer.resolved_object.do_lower_case.numpy()

    return model, vocab, do_lowercase


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
        self.x_train = None
        self.y_train = None
        self.x_test = None

        # Initialize model
        self.model, vocab, do_lowercase = get_bert_classifier(metadata['class_num'],
                                                              metadata['language'])
        self.tokenizer = FullTokenizer(vocab, do_lowercase)
        self.input_mask = tf.zeros((MAX_SEQ_LENGTH,))
        self.segment_ids = tf.zeros((MAX_SEQ_LENGTH,))
        

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
        if self.x_train is None:
            # If the preprocessed training data is not cached, preprocess it
            x_train, y_train = train_dataset
            x_train = preprocess_examples(x_train, self.tokenizer, self.metadata['language'])
            self.x_train, self.y_train = x_train, y_train

        # Train model
        print(self.input_mask.shape)
        print(self.segment_ids.shape)
        history = self.model.fit(
                    x=[self.x_train, self.input_mask, self.segment_ids],
                    y=self.y_train,
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
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        # Preprocess data
        x_test = preprocess_examples(x_test, self.tokenizer, metadata['language'])

        # Evaluate model
        return self.model.predict_classes([x_test, self.input_mask, self.segment_ids])
