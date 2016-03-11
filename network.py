import numpy as np
import pandas as pd

from keras.models import Graph
from keras.layers.recurrent import LSTM
from keras.layers.core import AutoEncoder, Dense, Activation, TimeDistributedDense, Flatten, Dropout, RepeatVector
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

    def __init__(self):
        self._model = self.get_model()


    def get_model(self):
        # TODO: this is broken and frustrating, come back to it
        model = Graph()
        model.add_input(name='input', dtype='int32') #embedding needs int32

        # Encoding input into a representation
        model.add_node(Embedding(input_dim=vocabulary_size, output_dim=embedding_size),
                       name='embedding',
                       input='input')
        model.add_node(LSTM(input_dim=embedding_size, output_dim=encoding_size, return_sequences=True),
                       name='encoder',
                       input='embedding')

        # Branch 1: reconstruct
        model.add_node(LSTM(input_dim=encoding_size, output_dim=decoding_size, return_sequences=True),
                       name='decoder',
                       input='encoder')
        model.add_node(TimeDistributedDense(input_dim=decoding_size, output_dim=vocabulary_size, activation='softmax'),
                       name='distributed',
                       input='decoder')
        model.add_output(name='reconstruction', input='distributed')

        # Branch 2: classify
        model.add_node(TimeDistributedDense(input_dim=encoding_size, output_dim=1),
                       name='transition',
                       input='encoder')
        model.add_node(Flatten(), name='transition2', input='transition')
        model.add_node(Dropout(0.5), name='dropout', input='transition2')
        model.add_node(Dense(input_dim=sequence_length, output_dim=1, activation='sigmoid'),
                       name='dense',
                       input='dropout')
        model.add_output(name='classification', input='dense')

        return model


    def predict(self, x):
        # assume x is a single point, for online/active purposes
        pass


    def update_weights(self, x, y=None):
        # update weights for x
        # if no label y, no supervision
        # if label, the classification objective and the query objective both get updates
        pass

