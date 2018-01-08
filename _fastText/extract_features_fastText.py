#!/usr/bin/env python
import os, sys
import codecs

def main():
  langs_dict = {}; id = 1
  list_of_files = sys.argv[1]
  base = os.path.basename(list_of_files)
  fname, ext = os.path.splitext(base)

  with codecs.open(list_of_files) as f, codecs.open("{}-meta.tsv".format(fname), mode="w", encoding="utf-8") as meta:
    print("lang\tfile", file=meta)
    for line in f:
      #TODO(bzz): check 'linesNum' feature, skip big files
      lang, path, linesNum = line.split(";")
      # pre-process/extract features in FastText format
      base = os.path.basename(path)
      filename, ext = os.path.splitext(base)
      file_content = ""
      with codecs.open(path.strip(), encoding="utf-8") as src_f:
        try:
          file_content = src_f.read().replace('\n', "\\n")
          #TODO(bzz): add other pre-processing
          #  
          #  insert " " for [],.!>
          print("{}|__label__{} {}".format(path.replace("/xxx/repos/", ""),lang,  file_content))
        except Exception as e:
          print("Failed to process {}, {}".format(path, e))
  pass

if __name__ == '__main__':
  main()
