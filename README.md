# Machine Learning for programming language identification

*Language identification* is the task of determining the languege, by looking on sample of text.

Goals:
 - build multiple models for identification of programming languages for a given file
 - compare accuracy and performance

Non-goal: ignore vendored&generated code, override \w user settings


## Spec/Requiremtns

 IN: file name, content(sample?) of the file
OUT: class(prose, documentation, code), programming language name

Go bindings. Inference should be possible from Golang code


## Plan

 0. [x] collect data
 1. [x] vowpal wabbit
 2. [x] fastText
 3. move to scikit-learn:
   - binary classificartion + plot AUC
   - decidion trees
 4. move to TF
   - feed-forward NN
   - use predictions from Golang
   - RNN
 6. move to PyTourch?


## Depencencies

```
brew install rebenv ruby-build cmake icu4c
rbenv install 2.4.2
rbenv global 2.4.2
rbenv version
ruby --verion
gem install bundler

LDFLAGS="-L/usr/local/opt/icu4c/lib" CPPFLAGS="-I/usr/local/opt/icu4c/include" gem install github-linguist

brew install jq q vowpal-wabbit

git clone 
pushd fasttext && make -j4 && popd
```

## Collect the data

 - Clone list of projects
 - for each: run linguist and annotate every file
 - Separate train/test datasets


Experiments

```
linguist --json | jq 'keys[]'

linguist --json | jq '."Objective-C"'

linguist --json | jq -r 'keys[] as $k | "\($k); \(.[$k][])"' | less

linguist --json | jq --arg pwd "$PWD" -r 'keys[] as $k | "\($k);\(.[$k][])"' | awk -F';' -v pwd=$PWD '{print $1 ";" pwd "" $2}' > files.csv
```


## Train

 read and vectorize input:
   - filename
   - file extension
   - shebang
   - 1-gram
   - 2-gram
   - 3-gram
   - words/tokens
   - bytes/integers


## Running

```
# collec data, get languages
./clone_and_annotate_each_file.sh

# stats: lang, number of lines, number of files
q -d";" "SELECT c1, SUM(c3) as s, COUNT(1) as cnt FROM ./annotated_files.csv GROUP BY c1 ORDER BY cnt DESC"
q -d";" "SELECT c1, SUM(c3) as s, COUNT(1) as cnt FROM ./annotated_files.csv GROUP BY c1 ORDER BY s DESC"

```

### Vowpal Wabbit

> OAA multiclass classification \w logistic regression

https://github.com/JohnLangford/vowpal_wabbit/wiki/One-Against-All-(oaa)-multi-class-example

Features:
 - file ext
 - file name

```
# extract features, convert to https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format#input-format
./extract_features_vw.py ./annotated_files.csv

# shuffle
./extract_features_vw.py ./annotated_files.csv | perl -MList::Util=shuffle -e 'print shuffle(<>);' > train_features.vw

# split train/validate
python split.py train_features.vw train_split.vw test_split.vw -p 0.8 -r popa2

# train, for 19 languages
vw -d train_split.vw --oaa 19 --loss_function logistic -f trained_vw.model

# individual prediction
vw -t -i trained_vw.model

# test
vw -t -i trained_vw.model test_split.vw -p test_split.predict

# P@1, R@1, AUC using http://osmot.cs.cornell.edu/kddcup/software.html
vw -d test.data -t -i trained_vw.model -r /dev/stdout | perf -roc -files gold /dev/stdin

# AUC using https://github.com/zygmuntz/kaggle-amazon/blob/master/auc.py
pip install ml_metrics
wget https://raw.githubusercontent.com/zygmuntz/kaggle-amazon/master/auc.py
python auc.py test_split.vw test_split.predict

 > AUC: 0.0614430665163
```

### fastText

From https://arxiv.org/abs/1607.01759

> linear models with a rank constraint and a fast loss approximation
> trains \w stochastic gradient descent and a linearly decaying learning rate
> CBOW-like, n-grams using the 'hashing trick'

https://github.com/facebookresearch/fastText#text-classification
https://github.com/facebookresearch/fastText/blob/master/tutorials/supervised-learning.md#getting-and-preparing-the-data

Features:
 - full text

```
# format input from `annotated_files.csv` to `__label__N <token1> <token2> ...`
./extract_features_fastText.py annotated_files.csv | perl -MList::Util=shuffle -e 'print shuffle(<>);' > repos-files.txt

# split
python split.py repos-files.txt repos-files.train repos-files.valid -p 0.7 -r dupa

#or
wc -l repos-files.txt
head -n 3000 repos-files.txt > repos-files.train
tail -n 1221 repos-files.txt > repos-files.valid

# train
fasttext supervised predict trained_fastText.model.bin -

# individual predictions
fasttext predict trained_fastText.model.bin -

# test + P@1, R@1 
fasttext test trained_fastText.model.bin repos-files.valid

N	1217
P@1	0.892
R@1	0.892

```



## TODO
 - make `clone_and_annotate_each_file.sh` pull, if repo in `./repos` alreayd exists
 - parallelize data collection \w GNU parallel or [equivalent](https://github.com/mmstick/parallel)
 - plot AUC http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html