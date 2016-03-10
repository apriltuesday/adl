"""
@author rkempter

Helper methods to work with the addressline data set and to generate bag
of word models
"""

import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from marvin.tokenizer import nltk_word_tokenizer


def generate_train_test_set(
        input_path, is_supervised=True,
        training_size=0.8, label_noise_size=0.1, seed=43):
    """
    If supervised:
        - 'training_size' of valid, resp. invalid records in training set
        - 1 - 'training_size' of valid, resp. invalid records in test set
    If Unsupervised:
        - 'training_size' of valid records are used for training
        - 1 - 'training_size' + invalid records are used for testing
        - Adjust for validation

    :param input_path: String, The records-file
    :param is_supervised: Boolean
    :param training_size: Float
    :param label_noise_size: (required if is_supervised=False) - amount of negative
                             examples in training set
    :param seed: Seed for random generator (default: 43)

    :return df_train, df_test: Panda dataframes
    """
    np.random.seed(seed)

    df = pd.read_csv(input_path)
    df.dropna()

    df_valid = df[df['label'] == 'VALID']
    df_invalid = df[df['label'] == 'INVALID']
    valid_count = len(df_valid)
    invalid_count = len(df_invalid)

    # Shuffle both datasets
    df_valid = df_valid.iloc[np.random.permutation(valid_count)]
    df_invalid = df_invalid.iloc[np.random.permutation(invalid_count)]

    invalid_percentage = training_size
    if not is_supervised:
        invalid_percentage = label_noise_size

    df_train = pd.concat([
        df_valid[:int(valid_count * training_size)],
        df_invalid[:int(invalid_count * invalid_percentage)]])

    df_test = pd.concat([
        df_valid[int(valid_count * training_size):],
        df_invalid[int(invalid_count * invalid_percentage):]
    ])

    return df_train, df_test


def get_bag_of_words_model(sentences, feature_count=1000):
    """
    Generates a bag of words model
    :param sentences: An iterable of strings
    :param feature_count: int, number of features used in bag of words model
    :return train_features, vectoriser: sparse feature matrix,
    trained CountVectorizer model
    """

    vectoriser = CountVectorizer(tokenizer=nltk_word_tokenizer,
                                 max_features=feature_count)
    train_features = vectoriser.fit_transform(sentences)

    return train_features, vectoriser


def replace_numbers_by_tokens(sentences, number_token='number'):
    """
    Replace all numbers \d+ by a token
    :param sentences: list of strings
    :param number_token: the token, numbers should be replaced with
    :return list of strings
    """

    sentences = [sentence.lower() for sentence in sentences]
    return [re.sub(r'\d+', number_token, s) for s in sentences]
