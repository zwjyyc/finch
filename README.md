## Contents
* [Machine Learning](https://github.com/zhedongzheng/finch#machine-learning)
* [Deep Learning](https://github.com/zhedongzheng/finch#deep-learning)
    * [Multilayer Perceptron](https://github.com/zhedongzheng/finch#multilayer-perceptron)
    * [Convolutional Neural Network](https://github.com/zhedongzheng/finch#convolutional-neural-network)
    * [Recurrent Neural Network](https://github.com/zhedongzheng/finch#recurrent-neural-network)
    * [Recurrent Convolutional Network](https://github.com/zhedongzheng/finch#recurrent-convolutional-network)
    * [Autoencoder](https://github.com/zhedongzheng/finch#autoencoder)
    * [Highway Network](https://github.com/zhedongzheng/finch#highway-network)
* [Computer Vision](https://github.com/zhedongzheng/finch#computer-vision)
    * [OpenCV in Python](https://github.com/zhedongzheng/finch#opencv-in-python)
* [Natural Language Processing](https://github.com/zhedongzheng/finch#natural-language-processing)
    * [Language Processing in Python](https://github.com/zhedongzheng/finch#language-processing-in-python)
* [Database](https://github.com/zhedongzheng/finch#database)
* [Distributed System](https://github.com/zhedongzheng/finch#distributed-system)
## Machine Learning
* Java &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;
#### Support Vector Machine
* [Theory](https://zhedongzheng.github.io/finch/svm.html)
* Java &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm_linear_clf.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm_linear_clf_test.py) &nbsp; | &nbsp; 
## Deep Learning
#### Multilayer Perceptron
* TensorFlow &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/mlp_clf_test.py) &nbsp; | &nbsp; 
#### Convolutional Neural Network
* TensorFlow &nbsp; | &nbsp; Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_1d_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf_test.py) &nbsp; | &nbsp;
#### Recurrent Neural Network 
 * TensorFlow &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_clf_cifar10_test.py) &nbsp; | &nbsp; 
 * TensorFlow &nbsp; | &nbsp; LSTM Regressor &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_regr.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_regr_test.py) &nbsp; | &nbsp; [Visualization](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp;
  * TensorFlow &nbsp; | &nbsp; LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
 * TensorFlow &nbsp; | &nbsp; LSTM Text Generation &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen.py) &nbsp; | &nbsp; [Shakespeare Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn_text_gen_test.py) &nbsp; | &nbsp; 
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
* [Word Embedding](https://github.com/zhedongzheng/finch/blob/master/natural-language-processing/word-embedding.ipynb)
## Database
* [SQL](https://github.com/zhedongzheng/finch/blob/master/database/postgresql.md)
## Distributed System
* [Spark Examples](https://github.com/zhedongzheng/finch/tree/master/spark/examples)
