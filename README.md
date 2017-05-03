![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)
All the code are written by myself. If you have any question or suggestion, you are more than welcome to send message to me.
## Contents
### Theory
* [Machine Learning](https://github.com/zhedongzheng/finch#machine-learning-theory)
* [Deep Learning](https://github.com/zhedongzheng/finch#deep-learning-theory)
* [Database System](https://github.com/zhedongzheng/finch#database-system-theory)
### Practice
* [Machine Learning](https://github.com/zhedongzheng/finch#machine-learning-practice)
  * [Linear Model](https://github.com/zhedongzheng/finch#linear-model)
  * [Support Vector Machine](https://github.com/zhedongzheng/finch#support-vector-machine)
  * [Decomposition](https://github.com/zhedongzheng/finch#decomposition)
* [Deep Learning](https://github.com/zhedongzheng/finch#deep-learning-practice)
  * [Multilayer Perceptron](https://github.com/zhedongzheng/finch#multilayer-perceptron)
  * [Convolutional Network](https://github.com/zhedongzheng/finch#convolutional-network)
  * [Recurrent Network](https://github.com/zhedongzheng/finch#recurrent-network)
  * [Recurrent Convolutional Network](https://github.com/zhedongzheng/finch#recurrent-convolutional-network)
  * [Autoencoder](https://github.com/zhedongzheng/finch#autoencoder)
  * [Highway Network](https://github.com/zhedongzheng/finch#highway-network)
* [Computer Vision](https://github.com/zhedongzheng/finch#computer-vision-practice)
  * [Basic Operations](https://github.com/zhedongzheng/finch#basic-operations)
  * [Image Segmentation](https://github.com/zhedongzheng/finch#image-segmentation)
  * [Detection](https://github.com/zhedongzheng/finch#detection)
* [Natural Language Processing](https://github.com/zhedongzheng/finch#natural-language-processing-practice)
* [Distributed System](https://github.com/zhedongzheng/finch#distributed-system-practice)
* [Web Framework](https://github.com/zhedongzheng/finch#web-framework-practice)
## Machine Learning (Theory)
* [Cross Entropy](https://zhedongzheng.github.io/finch/cross-entropy)
* [Support Vector Machine](https://zhedongzheng.github.io/finch/svm.html)
## Deep Learning (Theory)
* [Convolutional Network](https://zhedongzheng.github.io/finch/conv.html)
* [Recurrent Network](https://zhedongzheng.github.io/finch/rnn.html)
## Database System (Theory)
* [SQL Basics](https://github.com/zhedongzheng/finch/blob/master/database/postgresql.md)
## Machine Learning (Practice)
#### Linear Model
* TensorFlow &nbsp; | &nbsp; Linear Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_regr.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_regr_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/logistic.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/logistic_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;
#### Support Vector Machine
* TensorFlow &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm_linear_clf.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm_linear_clf_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp; 
#### Decomposition
* TensorFlow &nbsp; | &nbsp; Non-negative Matrix Factorization &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/nmf.py) &nbsp; | &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/nmf_movielens_test.py) &nbsp; | &nbsp;
## Deep Learning (Practice)
#### Multilayer Perceptron
* TensorFlow &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/mlp_clf_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
#### Convolutional Network
* TensorFlow &nbsp; | &nbsp; Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_1d_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Network
* TensorFlow &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf_cifar10_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Time Series Regressor &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_regr.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_regr_test.py) & [Visualization](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Text Generation &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen.py) &nbsp; | &nbsp; [Shakespeare Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen_sh_test.py) &nbsp; | &nbsp; [Nietzsche Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen_ni_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Convolutional Network
* TensorFlow &nbsp; | &nbsp; Recurrent Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
#### Autoencoder
* TensorFlow &nbsp; | &nbsp; Basic Model &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder_mnist_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Weights-tied Model &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder_tied_w.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder_tied_w_mnist_test.py) &nbsp; | &nbsp; 
#### Highway Network
* TensorFlow &nbsp; | &nbsp; Highway MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Highway CNN Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_conv_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_conv_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_conv_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp;
## Computer Vision (Practice)
#### Basic Operations
  * [Resize](https://github.com/zhedongzheng/finch/blob/master/computer-vision/resize.ipynb)
  * [Rotations](https://github.com/zhedongzheng/finch/blob/master/computer-vision/rotations.ipynb)
#### Image Segmentation
  * [Contours](https://github.com/zhedongzheng/finch/blob/master/computer-vision/contours.ipynb)
  * [Sorting Contours](https://github.com/zhedongzheng/finch/blob/master/computer-vision/sorting-contours.ipynb)
#### Detection
  * [Face & Eye Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/computer-vision/face-eye-detection.ipynb)
  * [Walker & Car Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/computer-vision/car-walker-detection.ipynb)
## Natural Language Processing (Practice)
* [Text Preprocessing](https://github.com/zhedongzheng/finch/blob/master/natural-language-processing/text-preprocessing.ipynb)
* [Word Indexing](https://github.com/zhedongzheng/finch/blob/master/natural-language-processing/word-indexing.ipynb)
## Distributed System (Practice)
* [Java Multithreading Example](https://github.com/zhedongzheng/finch/tree/master/java/MessageSwitchApp)
* [Spark Basic Examples](https://github.com/zhedongzheng/finch/tree/master/spark/examples)
## Web Framework (Practice)
* [Django Example - Dynamically Updating Stock Image](https://github.com/zhedongzheng/finch/tree/master/web/web_interface)
