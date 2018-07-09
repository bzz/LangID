#!/usr/bin/env python

"""
Simple feed-forward DNN model \w Keras API
"""

import os, sys
import csv
import argparse
import signal
from time import time
from glob import glob

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard


from vectorize import read_dict, snippetToVec
"""
Given dictionaris in
 - dict.txt
 - labels.txt

./model_linear.py -m ./model train snippets_enry_train_vec.csv
./model_linear.py -m ./model test snippets_enry_test_vec.csv
./model_linear.py predict < "println("hello world");"
"""

TRAIN_FILE="../_tf-estimator/snippets_enry_train_vec.csv"
TEST_FILE="../_tf-estimator/snippets_enry_test_vec.csv"
K_MODEL="keras-model-{}"

maxlen = 400 #TODO: change to 40-100
embedding_size = 50
batch_size = 64
epochs = 5

def load_data(filename):
    df = pd.read_csv(filename, sep=";", quoting=csv.QUOTE_NONE, names=['label', 'snippet'])
    return df['snippet'].map(lambda line: line.split(",")), df['label'].values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Mode: ", choices=("train", "test", "predict", "print-sentence-vectors", "visualize-snippet-vectors", "export"))
    parser.add_argument("input_file", help="Training data in CSV format")
    parser.add_argument("-d", "--dict", type=str, default="dict.txt", help="file to read a dictionary from")
    parser.add_argument("-l", "--labels-dict", type=str, default="labels.txt", help="load labels dictionary from")
    parser.add_argument("-m", "--model-dir", type=str, default="./model", help="save final model to")
    parser.add_argument("--meta", help="Visualization: path to the output file \w metadata")
    parser.add_argument("--doc", help="Visualization: path to the output file \w snippet vectors/embeddings")

    args = parser.parse_args()

    word_to_index = read_dict(args.dict)
    model_dir = os.path.join(args.model_dir, 'bow_embeddings_keras')

    if args.mode == "train":
        label_to_index = read_dict(args.labels_dict, 0)
        train(model_dir, len(word_to_index), len(label_to_index))
    elif args.mode == "test":
        label_to_index = read_dict(args.labels_dict, 0)
        test(model_dir, len(label_to_index))
    elif args.mode == "predict":
        predict(model_dir, word_to_index, label_to_index, args)
    elif args.mode == "print-snippet-vectors":
        print_snippet_vectors(model_dir, word_to_index, args)
    elif args.mode == "visualize-snippet-vectors":
        visualize_snippet_vectors(model_dir, word_to_index, args)
    elif args.mode == "export":
        export_save_model(model_dir, args.input_file)

def train(model_dir, max_words, num_classes):
    ((x_train, y_train), (x_test, y_test)) = load_data_all(
        TRAIN_FILE, TEST_FILE)
    print('{} classes'.format(num_classes))

    print('Pad sequences (samples x maxlen)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print('Build model...')
    model = Sequential(name="snippet_vec")

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_words,
                        embedding_size,
                        input_length=maxlen)) #mask_zero=True

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=model_dir)

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[tensorboard],
            validation_data=(x_test, y_test))

    model_file = os.path.join(model_dir, K_MODEL.format(time()))
    print("Saving model to '{}'".format(model_file))
    model.save(model_file)

def test(model_dir, num_classes):
    print('Loading test data...')
    (x_test, y_test) = _load_data_for("test", TEST_FILE)

    print('Pad sequences (samples x maxlen)')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix')
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_test shape:', y_test.shape)

    model_file = _get_model_filename(model_dir)
    model = keras.models.load_model(model_file)
    score = model.evaluate(x_test, y_test,
                        batch_size=batch_size, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def predict(model_dir, word_to_index, label_to_index, args):
    model_file = _get_model_filename(model_dir)
    model = keras.models.load_model(model_file)

    input_ = sys.stdin if args.input_file == "-" else open(args.input_file, "r")
    labels = {v: k for k, v in label_to_index.items()}
    for line in input_:  # assume \n -> \\n in the CLI input
        if SHOULD_STOP:  # handle Ctrl+C
            break
        line = line.replace("\n", "")
        if line:
            x = np.array([np.array(snippetToVec(line, word_to_index))])
            print("\t'{}' -> {}".format(line, x))
            x = sequence.pad_sequences(x, maxlen=maxlen)
            predictions =  model.predict(x, batch_size=1)
            for prediction in predictions:
                #print("\t{}".format(prediction))
                top_n_probs = np.argpartition(prediction, -3)[-3:]
                for i in top_n_probs[np.argsort(prediction[top_n_probs])[::-1]]:
                    print("\t {}, {:.2f}".format(labels[i], float(prediction[i])))

    input_.close()

def print_snippet_vectors(model_dir, word_to_index, args):
    avg_layer_model = _get_avg_layer(model_dir)
    input_ = sys.stdin if args.input_file == "-" else open(args.input_file, "r")
    for line in input_:  # assume \n -> \\n in the CLI input
        if SHOULD_STOP:  # handle Ctrl+C
            break
        line = line.replace("\n", "")
        if line:
            x = np.array([np.array(snippetToVec(line, word_to_index))])
            print("\t'{}' -> {}".format(line, x))
            x = sequence.pad_sequences(x, maxlen=maxlen)
            sentence_vector = avg_layer_model.predict(x)
            print(".".join([str(w) for w in sentence_vector[0]]))

    input_.close()


def export_save_model(model_dir, output_dir):
    model_file = _get_model_filename(model_dir)
    print("Export model from '{}' to '{}' in SaveModel format".format(model_file, output_dir))

    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)
    K.set_learning_phase(0)  # all new operations will be in test mode from now on

    # serialize the model and get its weights, for quick re-building
    previous_model = keras.models.load_model(model_file)
    config = previous_model.get_config()
    weights = previous_model.get_weights()

    # re-build a model where the learning phase is now hard-coded to 0
    #from keras.models import model_from_config
    #model_from_config(config)
    new_model = Sequential.from_config(config)
    new_model.set_weights(weights)

    # TF part
    builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        { "code" : new_model.input }, {"prediction" : new_model.output}
    )

    builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature,
      },
      legacy_init_op=legacy_init_op)

    builder.save()


def visualize_snippet_vectors(model_dir, word_to_index, args):
    skipped_empty_snippets = 0
    avg_layer_model = _get_avg_layer(model_dir)
    with open(args.doc, 'w') as doc, open(args.meta, 'w') as meta:
        meta.write("Lang\tFile\n")
        df = pd.read_csv(args.input_file, sep=chr(255)*3, encoding="utf8", quoting=csv.QUOTE_NONE, names=['path', 'lang', 'snippet'])
        for row in df.itertuples():  # assume \n -> \\n done already
            if SHOULD_STOP:  # handle Ctrl+C
                break
            #import pdb; pdb.set_trace()
            if row.snippet:
                #write snippet vector
                snippet = str(row.snippet).replace("\n", "")
                x = np.array([np.array(snippetToVec(snippet, word_to_index))])
                x = sequence.pad_sequences(x, maxlen=maxlen)
                sentence_vector = avg_layer_model.predict(x)
                doc.write("{}\n".format("\t".join([str(w) for w in sentence_vector[0]])))
                #write meta
                meta.write("{}\t{}\n".format(row.lang, row.path.replace("/Users/alex/floss/learning-linguist/dataset-1/repos/","")))
            else:
                #print("{}: '{}' empty snippet".format(row.Index, row.path), file=sys.stderr)
                skipped_empty_snippets += 1
        print("Empty snippets: {} of {}".format(skipped_empty_snippets, len(df.index)))


def _get_avg_layer(model_dir):
    model_file = _get_model_filename(model_dir)
    model = keras.models.load_model(model_file)
    layer_name = 'global_average_pooling1d_1'
    return Model(inputs=model.input,
                outputs=model.get_layer(layer_name).output)

def _get_model_filename(model_dir):
    path = os.path.join(model_dir, K_MODEL.format("*"))
    files = glob(path)
    if files and len(files) > 0:
        first = files[0]
        print("Using model file: {}".format(first))
        return first
    else:
        print("No model files found: '{}'".format(path))
        return ""

def load_data_all(train, test):
    print('Loading data...')
    tr = _load_data_for("train", train)
    ts = _load_data_for("test", train)
    return (tr, ts)

def _load_data_for(purpose, src_file):
    (x, y) = load_data(src_file)
    print('{} {} snippets'.format(len(x), purpose))
    print('Average {} snippet length: {}'.format(purpose, np.mean(list(map(len, x)), dtype=int)))
    return (x, y)


SHOULD_STOP = False  # Handle Ctrl+C gracefully
def _sigint_handler(signal, frame):
    global SHOULD_STOP
    SHOULD_STOP = True
signal.signal(signal.SIGINT, _sigint_handler)

if __name__ == "__main__":
    main()