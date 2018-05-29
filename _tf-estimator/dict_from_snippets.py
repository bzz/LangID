#!/usr/bin/env python
import sys
import collections
import argparse
import signal

"""
Build dictionaries from CSV file \w snippets

Input:  .csv \w columns: abs_path, lang, snippet (\n->\\n) on STDIN
Output: 
 - .txt \w dictiorany for all LoCs on STDOUT
 - .txt \w label dictionary to file
"""

SEPARATOR = chr(255)*3
MAX_DICT_SIZE = 400000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dict", type=str, default=None, help="save words dictionary to")
    parser.add_argument("-l", "--labels-dict", type=str, default=None, help="save labels dictionary to", required=True)
    parser.add_argument("-s", "--dict-size", type=int, default=MAX_DICT_SIZE, help="dictionary size")
    args  = parser.parse_args()

    print("Building dictionary", file=sys.stderr)

    processed, skipped = 0, 0
    input = sys.stdin
    word_output = sys.stdout if not args.dict else open(args.dict, 'w', encoding="utf8")
    word_count, label_count = collections.Counter(), collections.Counter()
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
        word_count.update(tokenized(chunk.replace("\\n", "\n")))
        label_count.update([lang.strip()])

    print("Lables: {} uniq".format(len(label_count)), file=sys.stderr)
    if args.labels_dict:
        print("Saving lables to {}".format(args.labels_dict), file=sys.stderr)
        with open(args.labels_dict, 'w', encoding="utf8") as lable_output:
            for label, _ in label_count.most_common():
                print(label, file=lable_output)
    else:
        print("Skip saving the lables as no CLI args provided", file=sys.stderr)

    dict_size = 0
    predefined = [('BOS', 1), ('EO', 2), ('UNK', 3)] # any of this can duplicate later :/
    for token, _ in predefined:
        print(token, file=word_output)
        dict_size += 1

    for word, _ in word_count.most_common(args.dict_size - len(predefined)):
        if SHOULD_STOP: # handle Ctrl+C
            break
        print(word, file=word_output)
        dict_size += 1

    print("Text: {} uniq words, final dictionary size: {}".format(len(word_count), dict_size), file=sys.stderr)
    print("Samples processed: {}, skipped: {}".format(processed, skipped), file=sys.stderr)
    word_output.close()

def tokenized(line, sep=None):
    """Used in 
      - training: vocab bilding, vectorization
      - inference
    """
    #TODO: more pre-processing here
    # - "start/end of the sentence", "start/end of the word" marks
    # - n-grams
    result = []
    if line:
        result = line.split(sep=sep)
    return result

SHOULD_STOP = False # Handle Ctrl+C gracefully
def sigint_handler(signal, frame):
    global SHOULD_STOP
    SHOULD_STOP=True
signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

if __name__ == '__main__':
    main()
