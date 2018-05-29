#!/usr/bin/env python
import os, sys
import collections
import argparse
import signal
import shutil
import tempfile

import tensorflow as tf

"""
Converts CSV file \w vectorized snippets to TFRecords format

Input:  .csv \w columns: label_vec, snippet_ver on STDIN
Output: .tfrecords file
"""

SEPARATOR = ";"
SEPARATOR_VEC=","

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="file to save TFRecords to", required=True)
    args = parser.parse_args()

    skipped, processed = 0, 0
    input = sys.stdin
    print("Converting dataset to TFRecordFormat: {}".format(args.output))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tfrecords") as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        for line in input:
            if SHOULD_STOP: # handle Ctrl+C
                break
            try: # Input format validation
                label, snippet = line.split(SEPARATOR)
                processed += 1
            except ValueError as e:
                skipped += 1
                print("Can not parse line: '{}'".format(line.replace("\n","\\n")), file=sys.stderr)
                continue
            label_vec = int(label)
            snippet_vec = snippet.split(SEPARATOR_VEC)
            ex = tf.train.SequenceExample()
            ex.context.feature["len"].int64_list.value.append(len(snippet_vec))
            ex.context.feature["label"].int64_list.value.append(label_vec)
            tokens = ex.feature_lists.feature_list["tokens"]
            for s in snippet_vec:
                tokens.feature.add().int64_list.value.append(int(s))
            writer.write(ex.SerializeToString())
        writer.close()

        shutil.copy(fp.name, args.output)
        os.remove(fp.name)
    print("Samples processed: {}, skipped: {}".format(processed, skipped), file=sys.stderr)

SHOULD_STOP = False # Handle Ctrl+C gracefully
def sigint_handler(signal, frame):
    global SHOULD_STOP
    SHOULD_STOP=True
signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

if __name__ == '__main__':
    main()