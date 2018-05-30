#!/usr/bin/env python

import os
import argparse

import tensorflow as tf

from vectorize import read_dict

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

    classifier = tf.estimator.LinearClassifier(
        feature_columns=[column], model_dir=os.path.join(args.model_dir, 'bow_sparse'), n_classes=len(label_to_index))

    if args.mode == "train":
        train(args, classifier)
    elif args.mode == "test":
        test(args, classifier)
    elif args.mode == "predict":
        predict(args)

def train(args, classifier):
    if not args.model_dir:
        args.model_dir = tempfile.mkdtemp()
        print("Model will be saved to {}".format(args.model_dir))
    else:
        if not os.path.exists(args.model_dir):
            print("Path '{}' does not exist, will create it".format(args.model_dir))
            os.makedirs(args.model_dir)

    train_data = args.input_file

    print("\n## Training")
    # Training for 4000 steps means 128*4000 training examples (with the default batch size)
    # This is roughly equivalent to 5 epochs since the training dataset contains 92088 expls
    classifier.train(input_fn=lambda: input_fn(train_data), steps=4000)

    ## Evaluation
    train_eval_result = classifier.evaluate(input_fn=lambda: input_fn(train_data))
    print("Training set accuracy: {accuracy}".format(**train_eval_result))


def input_fn(filename):
    def parseTFRecord(ex):
        context_features = {
            "len": tf.FixedLenFeature((), dtype=tf.int64),
            "label": tf.FixedLenFeature((), dtype=tf.int64),
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature((), dtype=tf.int64),
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex, context_features=context_features, sequence_features=sequence_features)

        return {
            "snippet_vec": sequence_parsed["tokens"],
            "len": context_parsed["len"],
        }, context_parsed["label"]

    def expand(x, y):
        x["len"] = tf.expand_dims(tf.convert_to_tensor(x["len"]), 0)
        y = tf.expand_dims(tf.convert_to_tensor(y), 0)
        return x, y

    def deflate(x, y):
        x['len'] = tf.squeeze(x['len'])
        y = tf.squeeze(y)
        return x, y

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.shuffle(buffer_size=154000)
    dataset = dataset.map(parseTFRecord)
    dataset = dataset.map(expand)
    dataset = dataset.padded_batch(128, padded_shapes=({"snippet_vec": [None], "len": [1]}, [1]))
    dataset = dataset.map(deflate)
    dataset.repeat()
    return dataset.make_one_shot_iterator().get_next()


def test(args, classifier):
    test_data = args.input_file
    print("\n## Evaluating")
    test_eval_result = classifier.evaluate(input_fn=lambda: input_fn(test_data))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))

    #TODO: 
    # - PR-curves
    # - learning curves (multiple accuracies on single TF board chart)

def predict(args):
    pass


if __name__ == "__main__":
    main()