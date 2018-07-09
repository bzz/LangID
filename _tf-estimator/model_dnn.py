#!/usr/bin/env python

import os, sys
import argparse
import signal

import tensorflow as tf
import numpy as np

from vectorize import read_dict
from model_linear import train, test, predict, sigint_handler
"""
Given dictionaris in
 - dict.txt
 - labels.txt

./model_linear.py -m ./model train snippets_enry_train.tfrecords
./model_linear.py -m ./model test snippets_enry_test.tfrecords
./model_linear.py predict < "println("hello world");"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Mode: ", choices=("train", "test", "predict"))
    parser.add_argument("input_file", help="Training data in TFRectord format")
    parser.add_argument("-d", "--dict", type=str, default="dict.txt", help="file to read a dictionary from")
    parser.add_argument("-l", "--labels-dict", type=str, default="labels.txt", help="load labels dictionary from")
    parser.add_argument("-m", "--model-dir", type=str, help="save final model to")

    args = parser.parse_args()

    word_to_index = read_dict(args.dict)
    label_to_index = read_dict(args.labels_dict, 0)

    column = tf.feature_column.categorical_column_with_identity('snippet_vec', len(word_to_index))

    embedding_size = 50
    word_embedding_column = tf.feature_column.embedding_column(column, dimension=embedding_size)

    classifier = tf.estimator.DNNClassifier(
        hidden_units=[100],
        feature_columns=[word_embedding_column],
        model_dir=os.path.join(args.model_dir, 'bow_embeddings'),
        n_classes=len(label_to_index))

    if args.mode == "train":
        train(args, classifier)
    elif args.mode == "test":
        test(args, classifier)
    elif args.mode == "predict":
        predict(args, classifier, word_to_index, label_to_index)


SHOULD_STOP = False  # Handle Ctrl+C gracefully
signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

if __name__ == "__main__":
    main()