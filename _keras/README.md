This is implementation of simple DNN classifier model in Keras API.

Limitations:
 - no dynamic batch sizes: all vectorisze code snippets have same length
 - by default, uses .csv->np.array and not .tfrecords->Tensor as input 
   allows to benefit from Keras simulanios training/evalution but is slower

## Using .csv

```
pv snippets_enry_train.csv | ./vectorize.py -l labels.txt -d dict.txt > snippets_enry_train_vec.csv

pv snippets_enry_test.csv | ./vectorize.py -l labels.txt -d dict.txt > snippets_enry_test_vec.csv

# Train
./model_dnn_fixed.py -d ../_tf-estimator/dict.txt -l ../_tf-estimator/labels.txt train 1

# Evaluate
./model_dnn_fixed.py -d ../_tf-estimator/dict.txt -l ../_tf-estimator/labels.txt -m model-full test 1

# Predict
./model_dnn_fixed.py -d ../_tf-estimator/dict.txt -l ../_tf-estimator/labels.txt -m model-full predict -

# Print full snippet vectors/embeddings (analog of "fasttext print-sentence-vectors)
./model_dnn_fixed.py -d ../_tf-estimator/dict.txt -l ../_tf-estimator/labels.txt -m model-full print-snippet-vectors -

./model_dnn_fixed.py -d ../_tf-estimator/dict.txt -l ../_tf-estimator/labels.txt \
   -m model-full \
   --meta snippets_enry_test_meta.csv \
   --doc snippets_enry_test_doc.csv \
   visualize-snippet-vectors \
   ../_tf-estimator/snippets_enry_test.csv

# Export trained mode in SaveModel format
./model_dnn_fixed.py -m model-full -d ../_tf-estimator/dict.txt export ./saved_model_dir
saved_model_cli show  --dir saved_model_dir --tag_set serve --signature_def serving_default

#
bazel-bin/tensorflow/python/tools/freeze_graph \


# Inference from Golang
// https://www.tensorflow.org/versions/r1.9/install/install_go
curl -o "libtensorflow-cpu-$(go env GOOS)-x86_64-1.9.0-rc2.tar.gz" \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$(go env GOOS)-x86_64-1.9.0-rc2.tar.gz"
tar -xvzf "libtensorflow-cpu-$(go env GOOS)-x86_64-1.9.0-rc2.tar.gz"

go run main.go



```

## science-3
```
docker run -it -e LD_PRELOAD= -v /storage:/storage --name alex --privileged srcd/science bash
docker exec -it alex bash

scp model_dnn_fixed.py science-3:/storage/alex/langid
python3 ./model_dnn_fixed.py -m ./model-full train 1
scp science-3:/storage/alex/langid/model-full/bow_embeddings_keras/keras-model-1528711009.8389738 model-full/bow_embeddings_keras
```


 ## Using TFRecords
https://github.com/keras-team/keras/blob/master/examples/mnist_dataset_api.py#L72

Keras can do evaluaiton while training, but not in this case of Tensor input.

# Further directions

## Dataset
 * add UNK category to training data
 * balance of classes
 * FAIR tools
 * add ngram features
 * train on MD snippets, dataset-3

## Model
 * [x] save model for re-use (hp5 format)
 * [ ] extract `common.py` \w vocab reading, to simplify imports
 * [ ] export model 1: SaveModel
       https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#exporting-a-model-with-tensorflow-serving

 * [ ] export model 2: GraphDef + weights
       https://github.com/vmarkovtsev/BiDiSentiment/blob/master/train_model.py#L165
       

Comparison
 * PR-curves
   https://medium.com/@akionakas/precision-recall-curve-with-keras-cd92647685e1
 * Learning curves
   https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
 * visualize document embeddings: print-sentence-vectors
   https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer

Error analysis
 * confusion matrix
   https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb

Training
 * algorithms
 * LSTM

## Applicatoin
 * Golang inference
   https://github.com/vmarkovtsev/BiDiSentiment/blob/master/inference.go
   https://github.com/src-d/tensorflow-codelab/tree/master/cmd
   https://github.com/google-aai/tf-serving-k8s-tutorial/blob/master/keras_training_to_serving.ipynb
   http://vmarkovtsev.github.io/codelab-2018-aalborg

 * Web: js inference on frontend
   https://github.com/tensorflow/tfjs-converter

 * Web: tf.Serving for backend
   https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md
