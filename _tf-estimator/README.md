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
