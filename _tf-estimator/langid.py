import tensorflow as tf
import pandas as pd
import numpy as np
from tensorboard import summary as summary_lib

import collections
import tempfile
import codecs
import pickle
import shutil
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
SNIPPET_DATASET_FILE = 'snippets.tfrecords'
SNIPPET_DICT_FILE = 'snippets_dict'


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
    if not os.path.exists(SNIPPET_DATASET_FILE):
        print("File {} does not exist, generating it by reading {}".format(SNIPPET_DATASET_FILE, DATASET_FILE))
        saveAsTFRecord(snippetsToVec(filesToSnippets(dataset), word_to_index, label_to_index))
    else:
        print("{} exists, reading it".format(SNIPPET_DATASET_FILE))

    # TODO add ngram features
    # TODO train/test data split

    #build model
    print("\n## Training")
    print("Model will be saved at: {}".format(model_dir))
    column = tf.feature_column.categorical_column_with_identity('snippet_vec', len(word_to_index))

    classifier = tf.estimator.LinearClassifier(
        feature_columns=[column], model_dir=os.path.join(model_dir, 'bow_sparse'), n_classes=len(label_to_index))

    # TODO DNN model
    #embedded_text_feature_column = text_embedding_column()
    #estimator = tf.estimator.DNNClassifier(
    #    hidden_units=[500, 100],
    #    feature_columns=[embedded_text_feature_column],
    #    n_classes=len(labels),
    #    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # Training for 4000 steps means 128*4000 training examples (with the default batch size)
    # This is roughly equivalent to 5 epochs since the training dataset contains 92088 expls
    classifier.train(input_fn=train_input_fn, steps=4000)

    ## Evaluation
    print("\n## Evaluating")  # TODO: input_fn for evaluation (on train/test datasets)
    train_eval_result = classifier.evaluate(input_fn=train_input_fn)
    print("Training set accuracy: {accuracy}".format(**train_eval_result))

    #test_eval_result = classifier.evaluate(input_fn=test_input_fn)
    #print "Test set accuracy: {accuracy}".format(**test_eval_result)

    def load():
        xs, ys = np.array([]), []
        dataset = tf.data.TFRecordDataset(SNIPPET_DATASET_FILE)
        dataset = dataset.map(parseTFRecord)
        dataset = dataset.map(lambda x, y: (x["snippet_vec"], tf.expand_dims(tf.convert_to_tensor(y), 0)))
        dataset = dataset.padded_batch(92212, padded_shapes=([None], [1]))
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(tf.squeeze(y), len(label_to_index))))
        whole_dataset_tensors = tf.contrib.data.get_single_element(dataset)
        with tf.Session() as sess:
            whole_dataset_arrays = sess.run(whole_dataset_tensors)
        return whole_dataset_arrays

    print("\n## Preparing data to generate PR-curves")
    x_test, y_test = load()

    print("\n## Predicting")
    pred = np.stack([
        p['probabilities'] for p in classifier.predict(
            input_fn=tf.estimator.inputs.numpy_input_fn(x={"snippet_vec": x_test}, batch_size=64, shuffle=False))
    ])

    # Add a PR summary for each class, in addition to the summaries that the classifier write
    print("\n## Building PR-curves")
    tf.reset_default_graph()
    with tf.Session() as pr_sess:
        for cat in label_to_index.values():
            with tf.name_scope('label_%s' % cat):
            #    _, update_op = summary_lib.pr_curve_streaming_op(
                update_op = summary_lib.pr_curve(
                    'precision_recall',
                    predictions=pred[:, cat],
                    labels=y_test[:, cat].astype(bool),
                    num_thresholds=30)
        merged_summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(classifier.model_dir, 'eval'), pr_sess.graph)
        pr_sess.run(tf.local_variables_initializer())
        pr_summary, _ = pr_sess.run([merged_summary_op, update_op])
        writer.add_summary(pr_summary, global_step=0)
        writer.close()


########## Pre-processing #############


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


def train_input_fn():
    dataset = tf.data.TFRecordDataset(SNIPPET_DATASET_FILE)
    dataset = dataset.shuffle(buffer_size=90000)
    dataset = dataset.map(parseTFRecord)
    dataset = dataset.map(expand)
    #dataset = dataset.apply(
    #    tf.contrib.data.padded_batch_and_drop_remainder(
    dataset = dataset.padded_batch(128, padded_shapes=({"snippet_vec": [None], "len": [1]}, [1]))
    dataset = dataset.map(deflate)
    #dataset.repeat()
    return dataset.make_one_shot_iterator().get_next()


def saveAsTFRecord(snipptes_vec):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tfrecords") as fp:
        print("Convert dataset to TFRecordFormat: {}".format(fp.name))
        writer = tf.python_io.TFRecordWriter(fp.name)
        i = 0
        for snippet_vec, label_vec in snipptes_vec:
            i += 1
            ex = tf.train.SequenceExample()
            ex.context.feature["len"].int64_list.value.append(len(snippet_vec))
            ex.context.feature["label"].int64_list.value.append(label_vec)
            tokens = ex.feature_lists.feature_list["tokens"]
            for s in snippet_vec:
                tokens.feature.add().int64_list.value.append(s)
            writer.write(ex.SerializeToString())
        writer.close()
        shutil.copy(fp.name, SNIPPET_DATASET_FILE)
        os.remove(fp.name)
        print("{} snippets total".format(i))


def snippetsToVec(snippets, word_to_index, label_to_index):
    for snippet, label in snippets:
        snippet_vec = np.array([word_to_index[w] if w in word_to_index else 2 for w in tokenized(snippet)])
        label_vec = label_to_index[label]
        yield snippet_vec, label_vec


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def filesToSnippets(dataset):
    chunks_loc = 10
    for _, row in dataset.iterrows():
        with codecs.open(row['file'].strip(), encoding="ISO-8859-1") as full_content_utf8:
            for lines_chunk in grouper(full_content_utf8, chunks_loc):
                snippet = ''.join(str(line) for line in lines_chunk)
                yield snippet, row['lang']
    #            snippets_vec.append(snippet_vec)
    #            labels_vec.append(label_to_index[row['lang']])
    #assert len(snippets_vec) == len(labels_vec), "{} snippets, {} labels".format(len(snippets_vec), len(labels_vec))
    #return snippets_vec, labels_vec


########## Dict building #############


def build_dicts(filenames, labels):
    dict_file = SNIPPET_DICT_FILE + ".pickle"
    if os.path.exists(dict_file):
        print("{} exists, reading dictionaries".format(dict_file))
        word_dict = pickle.load(open(dict_file, "rb"))
        lable_dict = pickle.load(open(SNIPPET_DICT_FILE + "_labels.pickle", "rb"))
        uniq_words = 0
    else:
        print("File {} does not exist, generating it now".format(SNIPPET_DICT_FILE + ".pickle"))
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
        lable_dict = dict()
        for label, _ in label_count.most_common():
            lable_dict[label] = len(lable_dict)
        #print(lable_dict)
        uniq_words = len(word_count)
        pickle.dump(word_dict, open(SNIPPET_DICT_FILE + ".pickle", "wb"))
        pickle.dump(lable_dict, open(SNIPPET_DICT_FILE + "_labels.pickle", "wb"))
    return word_dict, uniq_words, lable_dict


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
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return word_count, dictionary, reverse_dictionary

"""
