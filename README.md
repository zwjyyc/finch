## Contents
* [Machine Learning](https://github.com/zhedongzheng/finch#machine-learning)
    * [Linear Model](https://github.com/zhedongzheng/finch#linear-model)
    * [Support Vector Machine](https://github.com/zhedongzheng/finch#support-vector-machine)
    * [Decomposition](https://github.com/zhedongzheng/finch#decomposition)
* [Deep Learning](https://github.com/zhedongzheng/finch#deep-learning)
    * [Multilayer Perceptron](https://github.com/zhedongzheng/finch#multilayer-perceptron)
    * [Convolutional Network](https://github.com/zhedongzheng/finch#convolutional-network)
    * [Recurrent Network](https://github.com/zhedongzheng/finch#recurrent-network)
    * [Recurrent Convolutional Network](https://github.com/zhedongzheng/finch#recurrent-convolutional-network)
    * [Autoencoder](https://github.com/zhedongzheng/finch#autoencoder)
    * [Highway Network](https://github.com/zhedongzheng/finch#highway-network)
* [Computer Vision](https://github.com/zhedongzheng/finch#computer-vision)
    * [OpenCV in Python](https://github.com/zhedongzheng/finch#opencv-in-python)
* [Natural Language Processing](https://github.com/zhedongzheng/finch#natural-language-processing)
    * [Language Processing in Python](https://github.com/zhedongzheng/finch#language-processing-in-python)
* [Database](https://github.com/zhedongzheng/finch#database)
* [Distributed System](https://github.com/zhedongzheng/finch#distributed-system)
* [Web Framework](https://github.com/zhedongzheng/finch#web-framework)
## Machine Learning
#### Linear Model
* TensorFlow &nbsp; | &nbsp; Elastic Net Regressor &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/elastic_net_regr.py) &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/elastic_net_regr_test.py) &nbsp; | &nbsp; Elastic Net Classifier &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/elastic_net_clf.py) &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/elastic_net_clf_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;
#### Support Vector Machine
* [Theory](https://zhedongzheng.github.io/finch/svm.html)
* TensorFlow &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm_linear_clf.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm_linear_clf_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp; 
#### Decomposition
* TensorFlow &nbsp; | &nbsp; Non-negative Matrix Factorization &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/nmf.py) &nbsp; | &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/nmf_movielens_test.py) &nbsp; | &nbsp;
## Deep Learning
#### Multilayer Perceptron
* TensorFlow &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/mlp_clf_test.py) &nbsp; | &nbsp; 
#### Convolutional Network
* [Theory](https://zhedongzheng.github.io/finch/conv.html)
* TensorFlow &nbsp; | &nbsp; Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_1d_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf_test.py) &nbsp; | &nbsp;
#### Recurrent Network
* [Theory](https://zhedongzheng.github.io/finch/rnn.html)
* TensorFlow &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf_cifar10_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Time Series Regressor &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_regr.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_regr_test.py) & [Visualization](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Text Generation &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen.py) &nbsp; | &nbsp; [Shakespeare Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen_sh_test.py) &nbsp; | &nbsp; [Nietzsche Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen_ni_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/rnn_clf_test.py) &nbsp; | &nbsp;
#### Recurrent Convolutional Network
* TensorFlow &nbsp; | &nbsp; Recurrent Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
#### Autoencoder
* TensorFlow &nbsp; | &nbsp; Basic Model &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder_mnist_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Weights-tied Model &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder_tied_w.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder_tied_w_mnist_test.py) &nbsp; | &nbsp; 
#### Highway Network
* TensorFlow &nbsp; | &nbsp; Highway MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Highway CNN Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_conv_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_conv_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/hn_conv_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp;
## Computer Vision
#### OpenCV in Python
* Basic Operations
  * [Resize](https://github.com/zhedongzheng/finch/blob/master/computer-vision/resize.ipynb)
  * [Rotations](https://github.com/zhedongzheng/finch/blob/master/computer-vision/rotations.ipynb)
* Image Segmentation
  * [Contours](https://github.com/zhedongzheng/finch/blob/master/computer-vision/contours.ipynb)
  * [Sorting Contours](https://github.com/zhedongzheng/finch/blob/master/computer-vision/sorting-contours.ipynb)
* Detection
  * [Face & Eye Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/computer-vision/face-eye-detection.ipynb)
  * [Walker & Car Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/computer-vision/car-walker-detection.ipynb)
## Natural Language Processing
#### Language Processing in Python
* [Text Preprocessing](https://github.com/zhedongzheng/finch/blob/master/natural-language-processing/text-preprocessing.ipynb)
* [Word Indexing](https://github.com/zhedongzheng/finch/blob/master/natural-language-processing/word-indexing.ipynb)
## Database
* [SQL Basics](https://github.com/zhedongzheng/finch/blob/master/database/postgresql.md)
## Distributed System
* [Java Multi-threading Example](https://github.com/zhedongzheng/finch/tree/master/java/MessageSwitchApp)
* [Spark Basic Examples](https://github.com/zhedongzheng/finch/tree/master/spark/examples)
## Web Framework
* [Django Example - Dynamically Generating Stock Image from Matplotlib](https://github.com/zhedongzheng/finch/tree/master/web/web_interface)
