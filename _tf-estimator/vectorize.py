#!/usr/bin/env python
import sys
import collections
import argparse
import signal

from dict_from_snippets import tokenized

"""
Vectorize snippets and labels from a CSV using given dictionary

Input:  
  - .csv \w columns: abs_path, lang, snippet (\n->\\n) on STDIN
  - path to dictionary file
Output: .csv \w columns: label_vec; snippet_vec
"""

SEPARATOR = chr(255)*3
SEPARATOR_VEC=","

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dict", type=str, help="file to read a dictionary from", required=True)
    parser.add_argument("-l", "--labels-dict", type=str, help="save labels dictionary to", required=True)
    args = parser.parse_args()

    skipped, processed = 0, 0
    input = sys.stdin
    word_to_index = read_dict(args.dict)
    label_to_index = read_dict(args.labels_dict, 0)
    for line in input:
        if SHOULD_STOP: # handle Ctrl+C
            break
        try: # Input format validation
            path, lang, chunk = line.split(SEPARATOR)
            processed += 1
        except ValueError as e:
            skipped += 1
            print("Can not parse line: '{}'".format(line.replace("\n","\\n")), file=sys.stderr)
            continue
        snippet_vec = snippetToVec(chunk.replace("\\n", "\n"), word_to_index)
        label_vec = label_to_index[lang.strip()]
        print_vec(snippet_vec, label_vec)
    print("Samples processed: {}, skipped: {}".format(processed, skipped), file=sys.stderr)

def read_dict(filename, offset=1):
    print("Reading dictionary from '{}'".format(filename), file=sys.stderr)
    word_to_index = dict()
    with open(filename, 'r', encoding="utf8") as f:
        for word in f:
            word_to_index[word.strip()] = len(word_to_index)+offset
        if offset:
            word_to_index[''] = 0 # reserve 0
    print("Done. Dictionary size: {}".format(len(word_to_index)), file=sys.stderr)
    return word_to_index

def snippetToVec(snippet, word_to_index):
    snippet_vec = [word_to_index[w] if w in word_to_index else 3 for w in tokenized(snippet)]
    return snippet_vec

def print_vec(snippet_vec, label_vec):
    print("{};{}".format(label_vec, SEPARATOR_VEC.join(str(x) for x in snippet_vec)))

SHOULD_STOP = False # Handle Ctrl+C gracefully
def sigint_handler(signal, frame):
    global SHOULD_STOP
    SHOULD_STOP=True
signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

if __name__ == '__main__':
    main()