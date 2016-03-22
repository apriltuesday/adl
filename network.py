import numpy as np
import pandas as pd

from keras.models import Graph
from keras.layers.core import AutoEncoder, Dense, TimeDistributedDense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

from helper import generate_train_test_set
from lstm_networks import generate_input_matrix, generate_target_vectors


class Network(object):
    """
    Semi-supervised active deep neural network.
    Built here for binary classification of sequence data (e.g. text),
    but should be generalizable.
    """

    def __init__(self, initial_data=None,
                 embedding_size=10,
                 encoding_size=30,
                 decoding_size=10,
                 dropout_rate=0.5):
        # Ideally we'd like to initialize things with some unlabelled data to train on
        # or alternatively initialize with word2vec pretrained embeddings

        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.decoding_size = decoding_size
        self.dropout_rate = dropout_rate
        # data_dependent params
        self.vocabulary_size = 196 # XXX if char level, can we set this before getting data?
        self.sequence_length = 90 # XXX This isn't really doable online
        
        self.model = self.get_model()
        self.unsup_mask = {'classification': 0}


    def get_model(self):
        model = Graph()
        model.add_input(name='input', input_shape=(self.sequence_length,), dtype=int)

        # Encoding input
        model.add_node(Embedding(self.vocabulary_size, self.embedding_size, input_length=self.sequence_length),
                       name='embedding', input='input')
        model.add_node(LSTM(self.encoding_size, input_length=self.sequence_length, return_sequences=True),
                       name='encoder', input='embedding')

        # Branch 1: reconstruct
        model.add_node(LSTM(self.decoding_size, return_sequences=True),
                       name='decoder', input='encoder')
        model.add_node(TimeDistributedDense(self.vocabulary_size, activation='softmax'),
                       name='distributed', input='decoder')
        model.add_output(name='reconstruction', input='distributed')

        # Branch 2: classify
        model.add_node(Flatten(), name='flatten', input='encoder')
        model.add_node(Dropout(self.dropout_rate), name='dropout', input='flatten')
        model.add_node(Dense(1, activation='sigmoid'),
                       name='dense', input='dropout')
        model.add_output(name='classification', input='dense')

        # Branch 3: query
        # same as branch 2 but a different "label"
        # weights should be init so that we always query the 1st pt

        model.compile(optimizer='rmsprop', loss={'reconstruction':'mse',
                                                 'classification': 'binary_crossentropy'})
        return model


    # forward pass
    def predict(self, x):
        # NB online learning can't have sequence padding in any way that makes sense
        # assume x is a single point, for online/active purposes
        return self.model.predict({'input': x})


    # backward pass
    def update_weights(self, x, y=None):
        # update weights for x
        # if no label y, no supervision
        # if label, the classification objective and the query objective both get updates
        target = generate_target_vectors(x, self.vocabulary_size)
        if y:
            self.model.fit({'input': x,
                            'reconstruction': target,
                            'classification': y})
        else:
            self.model.fit({'input': x,
                            'reconstruction': target,
                            'classification': 1},
                           sample_weight=self.unsup_mask)
