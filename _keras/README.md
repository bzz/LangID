This is implementation of simple DNN classifier model in Keras API.

Limitations:
 - no dynamic batch sizes: all vectorisze code snippets have same length
 - by default, uses .csv->np.array and not .tfrecords->Tensor as input 
   allows to benefit from Keras simultanios training/evalution but is slower

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
tar -xvzf "libtensorflow-cpu-$(go env GOOS)-x86_64-1.export

export LIBRARY_PATH="$LIBRARY_PATH:$(pwd)/lib"
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$(pwd)/lib"

go get github.com/tensorflow/tensorflow/tensorflow/go
go test github.com/tensorflow/tensorflow/tensorflow/go

echo "import numpy as np" | go run main.go ./saved_model_dir ./../_tf-estimator/dict.txt


# Inference \w TF.Serving and REST
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md#installing-from-source

tensorflow_model_server --port=9000 --rest_api_port=9001 --model_name=langid --model_base_path=./saved_model_dir

curl -X POST \
  http://localhost:9001/v1/models/langid:predict \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{
  "signature_name": "serving_default",
  "instances": [
     { "b64": "$(echo "import numpy as np" | base64)" }
   ]
}'

# Inference from JavaScript
// https://github.com/tensorflow/tfjs-converter

# pip install --upgrade tensorflowjs #fix keras version :/
pip install "git+git://github.com/tensorflow/tfjs-converter#egg=tensorflowjs_dev&subdirectory=python"

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='dense_1_1/Softmax' \
    --saved_model_tags=serve \
    ./saved_model_dir \
    ./saved_model_web_dir



"A deep tree-based model for software defect prediction" by Dam, Hoa Khanh, et al https://arxiv.org/abs/1802.00921 

Research supported by @Samsung that adapts Tree-LSTM to use AST of source code, builds ast2vec node embeddings & trained on static analysis tool results of @TizenProject

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
 * [x] export model 1: SaveModel
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
 * [x] Golang inference
   https://github.com/vmarkovtsev/BiDiSentiment/blob/master/inference.go
   https://github.com/src-d/tensorflow-codelab/tree/master/cmd
   https://github.com/google-aai/tf-serving-k8s-tutorial/blob/master/keras_training_to_serving.ipynb
   http://vmarkovtsev.github.io/codelab-2018-aalborg

 * [ ] Web: js inference on frontend
   https://github.com/tensorflow/tfjs-converter

 * [x] Web: tf.Serving for backend
   https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md
