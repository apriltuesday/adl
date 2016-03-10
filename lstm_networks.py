"""
Methods to work with LSTM Autoencoder networks

@author: rkempter
"""
import os
import string

from datetime import datetime

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import recall_score, precision_score

import numpy as np
import pandas as pd


def generate_input_matrix(input_list, max_length=None, feature_count=None):
    """
    Transform strings to a sequence of indices (input to embedding layer)
    :param input_list
    :return list of list of ints, int input_dimension
    """
    sequences = []

    for sample in input_list:
        integer_list = []

        for character in sample:
            char_index = ord(character.lower())
            if feature_count and char_index > feature_count:
                raise InvalidArgumentError("The variable feature_count is too small")

            integer_list.append(char_index)

        sequences.append(integer_list)

    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    input_dimension = (max([index for seq in sequences for index in seq]) + 1
                       if not feature_count else feature_count)

    return padded_sequences, input_dimension


def get_optimal_precision_recall_from_clusters(
        df_cluster, cluster_label='cluster', class_label='label',
        invalid_label='INVALID', valid_label='VALID', fscore_beta=1):
    """

    Args:
        df_clustered:

    Returns:

    """
    df_grouped = pd.DataFrame({'count': df_cluster.groupby([cluster_label, class_label]).size()}).reset_index()
    df = df_grouped.pivot(index='cluster', columns='label', values='count')
    df = df.fillna(0)
    ratio = df['INVALID'] / (df['VALID'] + df['INVALID'])
    sorted_ratio = ratio.sort_values(ascending=False)

    invalid_clusters = []
    best_fscore = 0.0
    true_y = [1 if x == invalid_label else 0 for x in df_cluster['label']]

    for cluster, ratio in sorted_ratio.iteritems():
        current_clusters = [cluster] + invalid_clusters
        pred_y = [1 if x in current_clusters else 0 for x in df_cluster[cluster_label]]
        recall = recall_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y)
        fscore = (1 + fscore_beta ** 2) * (precision * recall) / ((fscore_beta ** 2) * precision + recall)

        if fscore > best_fscore:
            invalid_clusters = current_clusters
            best_fscore = fscore
        else:
            print "Recall: ", recall
            print "Precision: ", precision
            print "Fscore: ", fscore
            break

    return best_fscore, invalid_clusters


def generate_target_vectors(matrix, input_dimension):
    """
    Generates one-hot vectors for sequences of samples based on indices
    :param matrix: 2d array, size (nb_samples, nb_timesteps)
    :param input_dimension: int, vocabulary size
    :return target_vectors: 3d tensor (nb_samples, nb_timesteps, input_dimension)
    """
    target_vectors = np.zeros((matrix.shape[0], matrix.shape[1], input_dimension))
    for sample_nbr, sample in enumerate(matrix):
        for index, char_nbr in enumerate(sample):
            target_vectors[sample_nbr, index, char_nbr] = 1

    return target_vectors


def transform_to_string_sequence(sequence):
    """
    Transform a sequence of vocabulary indices (one-hot vectors)
    to a string
    :param sequence: list of list of integers
    :return list: list of strings
    """
    return ["".join([chr(c) for c in seq if c > 0]) for seq in sequence]
