#!/usr/bin/env python
import os, sys
import pickle
from pprint import pprint

def main():
  langs_dict = {}; id = 1
  list_of_files = sys.argv[1]
  with open(list_of_files) as f: #TODO(bzz): check 'filesize' feature, skip big files
    for line in f:
      lang, path, linum = line.split(";")
      # build dict
      if lang not in langs_dict:
          langs_dict[lang] = id
          id += 1
      lang_id = langs_dict[lang]

      # pre-process/extract features in VW format
      base = os.path.basename(path)
      filename, ext = os.path.splitext(base)
      file_content = ""
      with open(path.strip()) as src_f:
          file_content = src_f.read().replace('\n', ' ')
          #TODO(bzz): insert " " for [],.!>

      print("__label__{} {}".format(lang_id, file_content))
  #pickle.dump(langs_dict, open("lang_dict.pickle", "wb"))
  with open('lang_dict.txt', 'wt') as out:
    pprint(langs_dict, stream=out)


  

if __name__ == '__main__':
  main()
