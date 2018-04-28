#!/usr/bin/env python
import os, sys
import codecs
import argparse
from itertools import zip_longest

parser = argparse.ArgumentParser()
parser.add_argument("files")
parser.add_argument("--chunks", type=int, default=0, help="number of lines")
args = parser.parse_args()

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

PREFIX="/Users/alex/floss/learning-linguist/repos/"

def main():
  with codecs.open(args.files) as f:
    if args.chunks > 0:
      print("Using {} lines per example".format(args.chunks), file=sys.stderr)
    for line in f:
      #TODO(bzz): check 'linesNum' feature, skip big files
      lang, path, linesNum = line.split(";")
      # pre-process/extract features in FastText format
      base = os.path.basename(path)
      filename, ext = os.path.splitext(base)
      file_content = ""
      with codecs.open(path.strip(), encoding="utf-8") as src_f:
        try:
          if args.chunks > 0:
            for lines_chunk in grouper(src_f, args.chunks):
              chunk = ''.join(str(i) for i in lines_chunk).replace('\n', "\\n")
              print("{}|{}|{}".format(path.replace(PREFIX, ""), lang, chunk))
          else:
            file_content = src_f.read().replace('\n', "\\n")
            #TODO(bzz): add other pre-processing
            #  insert " " for [],.!>
            print("{}|{}|{}".format(path.replace(PREFIX, ""), lang, file_content))
        except Exception as e:
          print("Failed to process {}, {}".format(path, e))
  pass

if __name__ == '__main__':
  main()
