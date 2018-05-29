#!/usr/bin/env python
import os, sys
import codecs
import signal
import argparse
from itertools import zip_longest

"""
Reads a files from the list and generate chunks.

Input:  .csv \w columns: lang, abs_path, size
Output: .csv \w columns: rel_path, lang, chunk 
"""

BASE_PATH="/Users/alex/floss/learning-linguist/repos/"

parser = argparse.ArgumentParser()
parser.add_argument("--chunks", type=int, default=10, help="number of lines")
args = parser.parse_args()

def main():
    header = sys.stdin.readline()
    for line in sys.stdin:
        if SHOULD_STOP: # handle Ctrl+C
            break
        try: # Input format validation
            lang, path, linesNum = line.split(";")
        except ValueError as e:
            print("Can not parse line: '{}'".format(line), file=sys.stderr)
            continue
        with codecs.open(path.strip(), encoding="ISO-8859-1") as src_f:
            try:
                if args.chunks > 0:
                    for lines_chunk in grouper(src_f, args.chunks):
                        chunk = ''.join(str(i) for i in lines_chunk).replace('\n', "\\n")
                        printline(path, lang, chunk)
                else:
                    #TODO(bzz): limit max read size
                    full_file_content = src_f.read().replace('\n', "\\n")
                    printline(path, lang, full_file_content)
            except Exception as e:
                print("Failed to process {}, {}".format(path, e))
                sys.exit(1)

def printline(path, lang, chunk):
    print((chr(255)*3).join([path.replace(BASE_PATH, ""), lang, chunk]))

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

SHOULD_STOP = False # Handle Ctrl+C gracefully
def sigint_handler(signal, frame):
    global SHOULD_STOP
    SHOULD_STOP=True
signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

if __name__ == '__main__':
  main()
