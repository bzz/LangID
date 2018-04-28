import tensorflow as tf
import pandas as pd
import numpy as np

import collections
import tempfile
import codecs
import sys
import os
from itertools import zip_longest
#from nltk.util import ngrams
""" 
### Training

- [x] ['lang','file', 'LoC'] -> dictionary, dictionary_label
- [x] ['lang','file', 'LoC'] + dictionary, dictionary_label -> (snippets_vec, labels_vec)

train/test split
(snippets_vec, labels_vec) -> (snippets_train, labels_train), (snippets_test, labels_test)

column = tf.feature_column.categorical_column_with_identity('snippet_vec', vocab_size)

classifier = tf.estimator.LinearClassifier(  
    feature_columns=[column], 
    model_dir=os.path.join(model_dir, 'bow_sparse'))

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=snippets_train, labels_train)
classifier.train(nput_fn=train_input_fn)

eval_input_fn = ...
classifier.evaluate()


### Inference

print_predictions(sentences, classifier):
"""

DATASET_ROOT = '/Users/alex/floss/learning-linguist/repos/'
pd.set_option('line_width', 120)
pd.set_option('max_colwidth', 60)

vocabulary_size = 100000
model_dir = tempfile.mkdtemp()


def main():
    #read data
    DATASET_FILE = '../dataset-1/annotated_files.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=';', names=['lang', 'file', 'LoC'])
    print("## Files")
    print(dataset.head().to_string(formatters={'file': lambda x: "{}".format(x.replace(DATASET_ROOT, ''))}))
    print("{} samples\n".format(len(dataset)))
    print("\n### Files / Lang")
    print(dataset.groupby(['lang'])['file'].count().sort_values(ascending=False))
    print("\n### LoC / Lang")
    print(dataset.groupby(['lang'])['LoC'].sum().sort_values(ascending=False))

    #dict
    print("\n## Build dictionary")
    word_to_index, uniq_words, label_to_index = build_dicts(dataset['file'], dataset['lang'])
    print("Text: {} uniq words, dictionary size: {} \n".format(uniq_words, len(word_to_index)))
    print("Lables: {} uniq\n".format(len(label_to_index)))

    #featurize: sparse BoW
    print("\n## Vectorize words and labels")
    snippets_vec, labels_vec = toBoW(dataset, word_to_index, label_to_index)
    """
    SNIPPETS_FILE="snippets.txt"
    # ./extract_features_tf.py --chunks 10 ../dataset-1/annotated_files.csv | perl -MList::Util=shuffle -e 'print shuffle(<>);' > snippets.txt
    snippets = pd.read_csv(SNIPPETS_FILE, sep='|', names=['file', 'lang', 'text'])
    print("\n## Code snippets")
    print(snippets.loc[:, ['lang','text']].head())
    print()
    print("{} samples\n".format(len(snippets)))
    print("\n### Samples / Lang")
    print(snippets.groupby(['lang'])['file'].count().sort_values(ascending=False))
    #TODO(bzz): report dataset stats - Samples/lang
    
    # Dictionary
    #count, dictionary, reverse_dictionary = build_dict(snippets)
    #print("Text: {} uniq words, dictionary size: {} \n".format(len(count), len(dictionary)))
    #lables = snippets.lang.uniq()
    #print("Lables: {} uniq\n".format(len(lables)))
    """

    #ng = ngrams(text,5)

    # TODO train/test split
    #train_df = ...

    #embedded_text_feature_column = text_embedding_column()

    #build model
    print("\n## Training")
    print("Model will be saved at: {}".format(model_dir))
    column = tf.feature_column.categorical_column_with_identity('snippet_vec', len(word_to_index))

    classifier = tf.estimator.LinearClassifier(
        feature_columns=[column], model_dir=os.path.join(model_dir, 'bow_sparse'))

    #estimator = tf.estimator.DNNClassifier(
    #    hidden_units=[500, 100],
    #    feature_columns=[embedded_text_feature_column],
    #    n_classes=len(labels),
    #    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    ## Training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(snippets_vec, labels_vec, num_epochs=None, shuffle=True)

    # Training for 4000 steps means 128*4000 training examples (with the default batch size)
    # This is roughly equivalent to 5 epochs since the training dataset
    # contains 92088 examples.
    classifier.train(
        input_fn=train_input_fn, steps=4000)

    ## Evaluation
    # TODO: input_fn for evaluation (on train/test datasets)
    train_eval_result = classifier.evaluate(input_fn=predict_train_input_fn)
    #test_eval_result = classifier.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    #print "Test set accuracy: {accuracy}".format(**test_eval_result)


def text_embedding_column():
    return


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def toBoW(dataset, word_to_index, label_to_index):
    chunks_loc = 10
    snippets_vec, labels_vec = [], []
    for _, row in dataset.iterrows():
        with codecs.open(row['file'].strip(), encoding="ISO-8859-1") as full_content_utf8:
            #import pdb; pdb.set_trace()
            for lines_chunk in grouper(full_content_utf8, chunks_loc):
                for line in lines_chunk:
                    snippets_vec.append([word_to_index[w] if w in word_to_index else 2 for w in tokenized(line)])
                labels_vec.append(label_to_index[row['lang']])
    assert len(snippets_vec) == len(labels_vec), "{} snippets, {} labels".format(len(snippets_vec), len(labels_vec))
    return {'snippets_vec': np.array(snippets_vec)}, np.array(labels_vec)


def build_dicts(filenames, labels):
    word_count, label_count = collections.Counter(), collections.Counter()
    for _, filename in filenames.iteritems():
        with codecs.open(filename.strip(), encoding="ISO-8859-1") as full_content_utf8:
            try:
                c = collections.Counter(tokenized(full_content_utf8.read()))
                word_count.update(c)
            except Exception as e:
                print("Failed to process file: {}\n\t{}".format(filename, e))

    count = [['EOS', 1], ['UNK', 2]]
    count.extend(word_count.most_common(vocabulary_size - 2))
    word_dict = dict()
    for word, _ in count:
        word_dict[word] = len(word_dict)
    #reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))

    label_count.update((label.strip() for _, label in labels.iteritems()))
    lable_dict = dict(label_count)
    return word_dict, len(word_count), lable_dict


def tokenized(line, sep=None):
    """Used in 
     - training: vocab bilding, vectorization
     - inference
    """
    #TODO: more pre-processing here
    #TODO: pre-processing & "start/end of the sentence", "start/end of the word" marks?
    result = []
    if line:
        result = line.split(sep=sep)
    return result


def build_dict(rows):
    word_count = collections.Counter()
    for i, row in rows.iterrows():
        c = collections.Counter(row.astype(str)['text'].split())
        word_count.update(c)
        del c
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(i / len(rows) * 50), i / len(rows) * 100), end='')
        sys.stdout.flush()
    print()

    count = [['UNK', -1]]
    count.extend(word_count.most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)


#    data = list()
#    unk_count = 0
#    for word in words:
#        if word in dictionary:
#            index = dictionary[word]
#        else:
#            index = 0  # dictionary['UNK']
#            unk_count = unk_count + 1
#        data.append(index)
#    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return word_count, dictionary, reverse_dictionary

if __name__ == "__main__":
    main()
""" Low-level TF API for dealing \w embeddings

    embeddings_var = tf.get_variable(
        initializer=tf.zeros([vocab_size + num_oov_buckets, embeddings_dim]),
        name=EMBEDDINGS_VAR_NAME,
        dtype=tf.float32)
    lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, num_oov_buckets=1, default_value=-1)
    ids = lookup_table.lookup(tokens)
    combined_embedding = tf.nn.embedding_lookup(params=embeddings_var, ids=ids)

"""
