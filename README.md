
# ML for programming language identification

Goals:
 - build multiple models for identification of programming languages for a give file
 - compare performance/accuracy

Non-goal: ignore vendored&generated code, override \w user settings

## Predict
 IN: file name, content(sample?) of the files
OUT: class(prose, documentation, code), programming language name
Go binding? inference in Go API of TF


## Plan

 1. vowpal wabbit (OAA binary classification \w logistic regression)
 2. fastText
 3. decidion trees
 4. NN
 5. RNN



## Train

 collect data: smaple files for each programming language (how many?)
 separate train/test

 read and vectorize input:
   - file extension
   - 1-gram
   - 2-gram
   - 3-gram
 labels = directory names

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

Experiments

```
linguist --json | jq 'keys[]'

linguist --json | jq '."Objective-C"'

linguist --json | jq -r 'keys[] as $k | "\($k); \(.[$k][])"' | less

linguist --json | jq --arg pwd "$PWD" -r 'keys[] as $k | "\($k);\(.[$k][])"' | awk -F';' -v pwd=$PWD '{print $1 ";" pwd "" $2}' > files.csv
```

## Running

```
./extract_files.sh

q -d";" "SELECT c1,COUNT(1) as cnt FROM ./files_all.csv GROUP BY c1 ORDER BY cnt DESC"

# extract features
./prep.py

# shuffle
./prep.py | perl -MList::Util=shuffle -e 'print shuffle(<>);' > train.vw

# split train/validate
python split.py train.vw train_v.vw test_v.vw -p 0.8 -r popa

x
# train
vw -d train_v.vw --oaa 19 --loss_function logistic -f vw_model


# predict
vw -d test_v.vw --loss_function logistic -i vw_model -t

```
