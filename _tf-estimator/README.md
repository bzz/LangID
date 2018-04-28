Model for programming langiage identifications using TF and Estimators frameworks


## Papers

"Learning Deep Structured Semantic Models for Web Search using Clickthrough Data"
https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
Huang et al. 2013, Microsoft

"Efficient Estimation of Word Representations in Vector Space"
Tomas Mikolov et al. 2013, Google


"Bag of Tricks for Efficient Text Classification"
Tomas Mikolov ... 2016, Facebook

"Enriching Word Vectors with Subword Information"
Tomas Mikolov ... 2016, Facebook

"Natural Language Processing with Small Feed-Forward Networks"
Slav Petrov ... 2017, Google



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
