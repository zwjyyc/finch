![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)

## Installation
The following command clones all the files (>200MB)
```
git clone https://github.com/zhedongzheng/finch.git
```
Any test file can be run directly, the huge contents below is used to index the model and test file
```
python xxxx_test.py
```
The main dependence is:
* [tensorflow >= 1.2.2](https://www.tensorflow.org/)
* [pytorch >= 0.20](http://pytorch.org/)
* [mxnet >= 0.11](https://mxnet.incubator.apache.org/)
* [scikit-learn](http://scikit-learn.org/)
* [openai-gym](https://github.com/openai/gym)
## Other Language Support
[Chinese (中文)](https://github.com/zhedongzheng/finch/blob/master/README-CH.md)
## Style of My Code
I write each model as a single Class with ```fit()``` and ```predict()``` methods (scikit-learn API style), and then write separate test files for different incoming data. All the code below follow this style.
## Contents
* [Machine Learning](https://github.com/zhedongzheng/finch/blob/master/README.md#machine-learning)
  * [Linear Model](https://github.com/zhedongzheng/finch/blob/master/README.md#linear-model)
  * [Non-Linear Model](https://github.com/zhedongzheng/finch/blob/master/README.md#non-linear-model)
  * [Ensemble](https://github.com/zhedongzheng/finch/blob/master/README.md#ensemble)
* [Deep Learning](https://github.com/zhedongzheng/finch/blob/master/README.md#deep-learning)
  * [Multilayer Perceptron](https://github.com/zhedongzheng/finch/blob/master/README.md#multilayer-perceptron)
  * [Convolutional Network](https://github.com/zhedongzheng/finch/blob/master/README.md#convolutional-network)
  * [Recurrent Network](https://github.com/zhedongzheng/finch/blob/master/README.md#recurrent-network)
  * [Autoencoder](https://github.com/zhedongzheng/finch/blob/master/README.md#autoencoder)
  * [Highway Network](https://github.com/zhedongzheng/finch/blob/master/README.md#highway-network)
  * [Generative Adversarial Network](https://github.com/zhedongzheng/finch/blob/master/README.md#generative-adversarial-network)
* [Reinforcement Learning](https://github.com/zhedongzheng/finch/blob/master/README.md#reinforcement-learning)
* [Natural Language Processing](https://github.com/zhedongzheng/finch/blob/master/README.md#natural-language-processing)
  * [Text Representation](https://github.com/zhedongzheng/finch/blob/master/README.md#text-representation)
  * [Text Classification](https://github.com/zhedongzheng/finch/blob/master/README.md#text-classification)
  * [Text Generation](https://github.com/zhedongzheng/finch/blob/master/README.md#text-generation)
  * [Sequence Labelling](https://github.com/zhedongzheng/finch/blob/master/README.md#sequence-labelling)
  * [Sequence to Sequence](https://github.com/zhedongzheng/finch/blob/master/README.md#sequence-to-sequence)
* [Information Retrieval](https://github.com/zhedongzheng/finch/blob/master/README.md#information-retrieval)
  * [Recommendation System](https://github.com/zhedongzheng/finch/blob/master/README.md#recommendation-system)
* [Computer Vision](https://github.com/zhedongzheng/finch/blob/master/README.md#computer-vision)
  * [OpenCV](https://github.com/zhedongzheng/finch/blob/master/README.md#opencv)

## Machine Learning
#### Linear Model
* TensorFlow &nbsp; | &nbsp; Linear Regression &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Logistic Regression &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Support Vector Machine &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf_test.py) &nbsp; | &nbsp;

* Java &nbsp; | &nbsp; Logistic Regression &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;

* Java &nbsp; | &nbsp; Support Vector Machine &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp;
#### Non-Linear Model
* TensorFlow &nbsp; | &nbsp; K Nearest Neighbors &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/knn.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/knn_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; K-Means &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/kmeans.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/kmeans_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Random Forest &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/forest.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/forest_mnist_test.py) &nbsp; | &nbsp;
#### Ensemble
* Python &nbsp; | &nbsp; Bagging &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Adaboost &nbsp; &nbsp; [Pseudocode](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.md) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Random Forest &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf_test.py) &nbsp; | &nbsp;

## Deep Learning
#### Multilayer Perceptron
* TensorFlow &nbsp; | &nbsp; MLP Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; MLP Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
#### Convolutional Network
* TensorFlow &nbsp; | &nbsp; Conv2D Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Conv2D Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* MXNet &nbsp; | &nbsp; Conv2D Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/cnn/cnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/cnn/cnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/cnn/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Network
* TensorFlow &nbsp; | &nbsp; LSTM Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM Regressor &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr_plot.py) &nbsp; &nbsp; [Preview](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; GRU Regressor &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_regr_plot.py) &nbsp; | &nbsp;

* MXNet &nbsp; | &nbsp; LSTM Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* MXNet &nbsp; | &nbsp; LSTM Regressor &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/rnn/rnn_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/rnn/rnn_regr_plot.py) &nbsp; | &nbsp;
#### Autoencoder
* TensorFlow &nbsp; | &nbsp; Stacked Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Denoising Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/denoising_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/denoising_ae_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Sparse Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/sparse_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/sparse_ae_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Variational Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/variational_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/variational_ae_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Stacked Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/mlp_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Denoising Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/denoising_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/denoising_ae_mnist_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Sparse Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/sparse_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/sparse_ae_mnist_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Variational Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/variational_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/variational_ae_mnist_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Conv2D Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/conv_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/conv_ae_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
#### Highway Network
* TensorFlow &nbsp; | &nbsp; MLP Highway Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv1D Highway Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_hn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_hn_text_clf_imdb_test.py) &nbsp; | &nbsp;
#### Generative Adversarial Network
* TensorFlow &nbsp; | &nbsp; DCGAN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan_mnist_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan_mnist_test.md) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; DCGAN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/gan/dcgan.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/gan/dcgan_mnist_test.py) &nbsp; | &nbsp;

* MXNet &nbsp; | &nbsp; DCGAN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/gan/dcgan.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/mxnet-models/gan/dcgan_mnist_test.py) &nbsp; | &nbsp;
## Reinforcement Learning
* TensorFlow &nbsp; | &nbsp; Policy Gradient &nbsp; &nbsp; [Model 1](https://github.com/zhedongzheng/finch/blob/master/rl-models/tensorflow/pg.py) &nbsp; &nbsp; [CartPole Test](https://github.com/zhedongzheng/finch/blob/master/rl-models/tensorflow/pg_cartpole_test.py) &nbsp; | &nbsp; [Model 2](https://github.com/zhedongzheng/finch/blob/master/rl-models/tensorflow/pg_2.py) &nbsp; &nbsp; [CartPole Test](https://github.com/zhedongzheng/finch/blob/master/rl-models/tensorflow/pg_2_cartpole_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Policy Gradient &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/rl-models/pytorch/pg.py) &nbsp; &nbsp; [CartPole Test](https://github.com/zhedongzheng/finch/blob/master/rl-models/pytorch/pg_cartpole_test.py) &nbsp; | &nbsp;

## Natural Language Processing
#### Text Representation
* Python &nbsp; | &nbsp; LSA &nbsp; | &nbsp; [Model for Visualization](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.md) &nbsp; | &nbsp; [Model for Concepts](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept_test.md) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; TF-IDF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Word Embedding - Skip-Gram &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Word Embedding - CBOW &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow_test.py) &nbsp; | &nbsp;
#### Text Classification
* Python &nbsp; | &nbsp; TF-IDF + LR &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_logistic.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv1D &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Multi-kernel Conv1D &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv1D-LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Conv1D &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Conv1D-LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_attn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* MXNet &nbsp; | &nbsp; Conv1D &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/mxnet/cnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/mxnet/cnn_text_clf_imdb_test.py) &nbsp; | &nbsp;
#### Text Generation
* Python &nbsp; | &nbsp; 2nd order Markov &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/markov_text_gen.py) &nbsp; &nbsp; [Robert Frost Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/markov_text_gen_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Char-RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen.py) &nbsp; | &nbsp; [《Anna Karenina》 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_test.py) &nbsp; &nbsp; [Beijing Address Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_addr_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Char-RNN (Beam Search) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_beam.py) &nbsp; &nbsp; [《Anna Karenina》 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_beam_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Char-RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_gen.py) &nbsp; | &nbsp; [《Anna Karenina》 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_gen_test.py) &nbsp; &nbsp; [Beijing Address Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_gen_addr_test.py) &nbsp; | &nbsp;

* MXNet &nbsp; | &nbsp; Char-RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/mxnet/rnn_text_gen.py) &nbsp; | &nbsp; [《Anna Karenina》 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/mxnet/rnn_text_gen_test.py) &nbsp; &nbsp; [Beijing Address Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/mxnet/rnn_text_gen_addr_test.py) &nbsp; | &nbsp;
#### Sequence Labelling
* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_seq2seq_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_rnn_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_rnn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Bi-directional LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_seq2seq_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Bi-directional LSTM + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_crf_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_crf_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_crf_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_seq_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_tagging_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_chseg_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Bi-directional LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_seq_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_tagging_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_chseg_test.py) &nbsp; | &nbsp;
#### Sequence to Sequence
* TensorFlow &nbsp; | &nbsp; Seq2Seq &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Seq2Seq + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Seq2Seq + Bi-directional Encoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Seq2Seq + Beam-Search Decoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Seq2Seq + Bi-directional Encoder + Attention + Beam-Search Decoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Seq2Seq &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/seq2seq.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/seq2seq_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Seq2Seq + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/seq2seq_attn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/seq2seq_attn_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Seq2Seq + Bi-directional Encoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/seq2seq_birnn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/seq2seq_birnn_test.py) &nbsp; | &nbsp;

## Information Retrieval
#### Recommendation System
* Python &nbsp; | &nbsp; Collborative Filtering &nbsp; | &nbsp; MovieLens &nbsp; &nbsp; [User-based Model](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/ncf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/ncf_movielens_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Apriori &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/apriori.py) &nbsp; &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/apriori_movielens_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Matrix Factorization &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/ir-models/tensorflow/nmf.py) &nbsp; &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/ir-models/tensorflow/nmf_movielens_test.py) &nbsp; | &nbsp;

## Computer Vision
#### OpenCV
* OP &nbsp; | &nbsp; [Resize](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/resize.ipynb)

* OP &nbsp; | &nbsp; [Rotations](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/rotations.ipynb)

* Segmentation &nbsp; | &nbsp; [Contours](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/contours.ipynb)

* Segmentation &nbsp; | &nbsp; [Sorting Contours](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/sorting-contours.ipynb)

* Segmentation &nbsp; | &nbsp; [Line detection](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/line-detection.ipynb)

* Segmentation &nbsp; | &nbsp; [Circle detection](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/circle-detection.ipynb)

* Segmentation &nbsp; | &nbsp; [Blob detection](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/blob-detection.ipynb)

* Detection &nbsp; | &nbsp; [Face & Eye Detection Using Cascade Classifier](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/face-eye-detection.ipynb)

* Detection &nbsp; | &nbsp; [Walker & Car Detection Using Cascade Classifier](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/car-walker-detection.ipynb)
