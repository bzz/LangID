#!/usr/bin/env python

# Preprocess filename.valid file for visualization though TF board
# Output:
# - <>-metadata.tsv
# - <>-docs.txt
#   fasttext print-sentence-vectors < <>-docs.txt | tr " " "\t" > snippets.tsv

import os, sys
import codecs

def main():
  docs_with_labels = sys.argv[1]
  base = os.path.basename(docs_with_labels)
  fname, ext = os.path.splitext(base)

  wcl = 0
  with codecs.open("{}-nolabel.txt".format(fname), mode="w", encoding="utf-8") as docs:
    with codecs.open("{}-meta.tsv".format(fname), mode="w", encoding="utf-8") as meta:
      print("{}\t{}".format("Lang", "File"), file=meta)
      with open(docs_with_labels) as f:
        for line in f:
          wcl+=1
          langPath = line[:line.find(' ')].strip().replace("__label__", "")
          lang = langPath[langPath.find('|')+1:]
          path = langPath[:langPath.find('|')]
          doc = line[line.find(' ')+1:].strip()
          print("{}".format(doc), file=docs)
          print("{}\t{}".format(lang, path), file=meta)
  print("{} lines processed".format(wcl))

if __name__ == '__main__':
  main()


#        try:
#        except Exception as e:
#          print("Failed to process {}, {}".format(path, e))
