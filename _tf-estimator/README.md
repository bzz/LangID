Model for programming langiage identifications using TF and Estimators frameworks


## Papers

### Microsoft, 2013 - DSSM
"Learning Deep Structured Semantic Models for Web Search using Clickthrough Data"
https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
Huang et al. 2013, Microsoft

> Given a word (e.g. good), we first add word starting and ending marks to the word (e.g. #good#).
> Then, we break the word into letter n-grams (e.g. letter trigrams: #go, goo, ood, od#)

### Facebook, 2016 - fastText
_"Bag of Tricks for Efficient Text Classification"_
https://arxiv.org/pdf/1607.01759.pdf
Tomas Mikolov ... 2016, Facebook

> we use a bag of n-grams as additional features to capture some partial information
> about the local word order
> The word representations are then _averaged_ into a text representation

_"Enriching Word Vectors with Subword Information"_
https://research.fb.com/wp-content/uploads/2017/06/tacl.pdf?
Tomas Mikolov ... 2016, Facebook

> bag of character n-gram. We add special boundary symbols < and >
> at the beginning and end of words, allowing to distinguish prefixes and suffixes
> a word is represented by its index in the word dictionary & set of hashed n-grams it contains
> <wh, whe, her, ere, re>, <where>
> We represent a word by the _sum_ of the vector representations of its n-grams


### Google, 2017 - CLD3
_"Natural Language Processing with Small Feed-Forward Networks"_
https://arxiv.org/abs/1708.00214
Slav Petrov ... 2017, Google

> we obtain the input separately _averaging_ the embeddings for each n-gram length (N = [1, 4]),
> as summation did not produce good results.

> We preprocess data by removing non-alphabetic characters and pieces of markup text
> (i.e., anything located between < and >, including the brackets)


## embeddings
 - general co-oc, unsupervised (word2vec)
 - task-specific, supervised   (labels instead of middle-word)

## tokenization?
text feature -> vector (both, train and inference)
 BoW
 BoNgrams
 BoW+BoNgrams

OneHot, but as list [w1, w1_ngram1, w1_ngram2, ..., w1_ngramN, w2, ...]


## vocabulary serialization
Important for re-use between training/inference.
https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py#L101


## Examples

Training/Evaluation in TF
```
./filter.py -o annotated_files_enry_filtered.csv --freq 100 ../dataset-1/annotated_files_enry.csv

pv annotated_files_enry_filtered.csv | ./snippets_from_files.py --chunks 10 \
    | perl -MList::Util=shuffle -E 'srand 123; print shuffle(<>);' \
    | ./split.py -p 0.8 -r 123 snippets_enry_train.csv snippets_enry_test.csv

pv snippets_enry_train.csv | ./dict_from_snippets.py -l labels.txt > dict.txt

pv snippets_enry_train.csv | ./vectorize.py -l labels.txt -d dict.txt | ./csv_to_tfrecords.py -o snippets_enry_train.tfrecords
pv snippets_enry_test.csv  | ./vectorize.py -l labels.txt -d dict.txt | ./csv_to_tfrecords.py -o snippets_enry_test.tfrecords

rm ../dataset-1/annotated_files_enry_filtered.csv snippets_enry_train.csv snippets_enry_test.csv dict.txt snippets_enry_train.tfrecords snippets_enry_test.tfrecords

train.py -d dict.txt -l labels.txt -o model.??? snippets_train.tfrecords
eval.py  -d dict.txt -m model.??? -o ./metrics_test snippets_test.tfrecords
tensorboard --logdir=./metrics_test

predict.py -d dict.txt -m model.???
```
#TODO
 - *.tfrecords sizes are x2 bigger :/
 - label_to_index is needed for evaluation



 - un-balanced split, some classes have not examples
 - "Gettext Catalog" is ok but the most frequent
 - "Modelica" for Django *.m   is wrong
 - "Roff"     for git/LGPL-2.1 is wrong


Inference in 
 - Golang https://github.com/src-d/tensorflow-codelab
 - JavaScript https://github.com/tensorflow/tfjs-converter

```
//quantilize  -m model.??? -o qmodel.???

//convert -d dict.txt -m qmodel.??? -o ./go_model
go build ./cmd/...
./predict

//convert -d dict.txt -m qmodel.??? -o ./js_model
python -m SimpleHTTPServer ./js_model
open http://locahost:8000/index.html
```


## Run
```
pip install pandas
```

text features: vocabulary -> tokenize ->
lables:

> python3 langid.py

 - Train
 - Test/Evaluate
 - Predict
