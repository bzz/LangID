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

 0. collect data
 1. vowpal wabbit (OAA multiclass classification \w logistic regression)
 2. fastText
 3. decidion trees
 4. NN
 5. RNN


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
q -d";" "SELECT c1, SUM(c3) as s, COUNT(1) as cnt FROM ./files_all.csv GROUP BY c1 ORDER BY cnt DESC"
q -d";" "SELECT c1, SUM(c3) as s, COUNT(1) as cnt FROM ./files_all.csv GROUP BY c1 ORDER BY s DESC"

# extract features
./prep.py

# shuffle
./prep.py | perl -MList::Util=shuffle -e 'print shuffle(<>);' > train.vw

# split train/validate
python split.py train.vw train_v.vw test_v.vw -p 0.8 -r popa

# train
vw -d train_v.vw --oaa 19 --loss_function logistic -f vw_model

# predict
vw -d test_v.vw --loss_function logistic -i vw_model -t

# AUC #1, using http://osmot.cs.cornell.edu/kddcup/software.html
vw -d test.data -t -i model.vw -r /dev/stdout | perf -roc -files gold /dev/stdin

# AUC #2, using https://github.com/zygmuntz/kaggle-amazon/blob/master/auc.py
pip install ml_metrics
wget https://raw.githubusercontent.com/zygmuntz/kaggle-amazon/master/auc.py
python pip.py <> <>

```

## TODO
 - make `clone_and_annotate_each_file.sh` pull, if repo in `./repos` alreayd exists
 - parallelize data collection \w GNU parallel or [equivalent](https://github.com/mmstick/parallel)