#!/usr/bin/env python
import os
import pickle
from pprint import pprint

def main():
  langs = {}; id = 1
  with open("./files_all.csv") as f: #TODO(bzz): check size, skip big files
    for line in f:
      # pre-process/extract features in VW format:
      # ext, filename, shebang #? 1grams, 2grams, 3grams
      lang, path, linum = line.split(";")
      if lang not in langs:
          langs[lang] = id
          id += 1
      lang_id = langs[lang]

      base = os.path.basename(path)
      filename, ext = os.path.splitext(base)
      shebang = ""
      #with open(path.strip()) as src_f:
      #    shebang = src_f.readline()
      print("{} | {} | {}".format(lang_id, ext.strip(), filename.strip())) #, shebang.strip()))
  pickle.dump(langs, open("lang_dict.pickle", "wb"))
  with open('lang_dict.txt', 'wt') as out:
    pprint(langs, stream=out)

  
  

if __name__ == '__main__':
  main()
