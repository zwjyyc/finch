![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)

This project contains my models, tests, and notes written on machine learning (ML) topics

Still at the early stage, many things need to be done in future

## Contents
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
  * [Generative Adversarial Network](https://github.com/zhedongzheng/finch#generative-adversarial-network)
* [Computer Vision](https://github.com/zhedongzheng/finch#computer-vision-practice)
  * [Basic Operations](https://github.com/zhedongzheng/finch#basic-operations)
  * [Image Segmentation](https://github.com/zhedongzheng/finch#image-segmentation)
  * [Detection](https://github.com/zhedongzheng/finch#detection)
  * [Deep CV](https://github.com/zhedongzheng/finch#deep-cv)
* [Natural Language Processing](https://github.com/zhedongzheng/finch#natural-language-processing-practice)
  * [Preprocessing](https://github.com/zhedongzheng/finch#preprocessing)
  * [Deep NLP](https://github.com/zhedongzheng/finch#deep-nlp)
### Notes
* Machine Learning
    * [Cross Entropy](https://zhedongzheng.github.io/finch/cross-entropy)
    * [Support Vector Machine](https://zhedongzheng.github.io/finch/svm.html)
* Deep Learning
    * [Network Optimization](https://zhedongzheng.github.io/finch/nn-optimization.html) 
    * Convolutional Network
        * [Overview](https://zhedongzheng.github.io/finch/conv.html)
        * [Pooling](https://zhedongzheng.github.io/finch/pooling.html)
    * Recurrent Network
        * [Recurrent Structure](https://zhedongzheng.github.io/finch/rnn.html)
        * [Long Short-Term Memory](https://zhedongzheng.github.io/finch/lstm.html)
        * [Variable Length Input Sequence](https://zhedongzheng.github.io/finch/rnn-practical.html)
## Machine Learning (Practice)
#### Linear Model
* TensorFlow &nbsp; | &nbsp; Linear Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;
#### Support Vector Machine
* TensorFlow &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp;
* LibSVM &nbsp; | &nbsp; Kernel SVM Classifier &nbsp; | &nbsp; [RBF Test](https://github.com/zhedongzheng/finch/blob/master/lib-tests/libsvm_clf_test.py) &nbsp; | &nbsp;
#### Decomposition
* TensorFlow &nbsp; | &nbsp; Non-negative Matrix Factorization &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/decomposition/nmf.py) &nbsp; | &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/decomposition/nmf_movielens_test.py) &nbsp; | &nbsp;
## Deep Learning (Practice)
#### Multilayer Perceptron
* TensorFlow &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; MLP Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_cifar10_test.py) &nbsp; | &nbsp;
* PyTorch &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
#### Convolutional Network
* TensorFlow &nbsp; | &nbsp; Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_1d_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
* PyTorch &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Network
* TensorFlow &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Regressor &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr.py) &nbsp; | &nbsp; [Sine Wave Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr_test.py) &nbsp; &nbsp; [Plot](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr_plot.py) &nbsp; &nbsp; [GIF](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Char Language Model &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen.py) &nbsp; | &nbsp; [Shakespeare Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen_sh_test.py) &nbsp; | &nbsp; [Nietzsche Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen_ni_test.py) &nbsp; | &nbsp; [Jay Chou Lyrics Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen_jay_test.py) &nbsp; | &nbsp; 
* PyTorch &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Convolutional Network
* TensorFlow &nbsp; | &nbsp; Conv1D-LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn_rnn/conv_rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn_rnn/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
#### Autoencoder
* TensorFlow &nbsp; | &nbsp; MLP Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_cifar10_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
#### Generative Adversarial Network
* TensorFlow &nbsp; | &nbsp; MLP Basic GAN &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan.py) &nbsp; | &nbsp; [Quadratic Drawing Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan_test.py) &nbsp; | &nbsp;
## Computer Vision (Practice)
#### Basic Operations
  * OpenCV &nbsp; | &nbsp; [Resize](https://github.com/zhedongzheng/finch/blob/master/computer-vision/resize.ipynb)
  * OpenCV &nbsp; | &nbsp; [Rotations](https://github.com/zhedongzheng/finch/blob/master/computer-vision/rotations.ipynb)
#### Image Segmentation
  * OpenCV &nbsp; | &nbsp; [Contours](https://github.com/zhedongzheng/finch/blob/master/computer-vision/contours.ipynb)
  * OpenCV &nbsp; | &nbsp; [Sorting Contours](https://github.com/zhedongzheng/finch/blob/master/computer-vision/sorting-contours.ipynb)
#### Detection
  * OpenCV &nbsp; | &nbsp; [Face & Eye Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/computer-vision/face-eye-detection.ipynb)
  * OpenCV &nbsp; | &nbsp; [Walker & Car Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/computer-vision/car-walker-detection.ipynb)
#### Deep CV
* TensorFlow &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
* PyTorch &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/torch-models/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;
## Natural Language Processing (Practice)
#### Preprocessing
* Python &nbsp; | &nbsp; [Text Cleaning](https://github.com/zhedongzheng/finch/blob/master/natural-language-processing/text-cleaning.ipynb)
* Python &nbsp; | &nbsp; [Word Indexing](https://github.com/zhedongzheng/finch/blob/master/natural-language-processing/word-indexing.ipynb)
#### Deep NLP
* TensorFlow &nbsp; | &nbsp; LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_1d_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Conv1D-LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn_rnn/conv_rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn_rnn/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Char Language Model &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen.py) &nbsp; | &nbsp; [Shakespeare Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen_sh_test.py) &nbsp; | &nbsp; [Nietzsche Texts Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen_ni_test.py) &nbsp; | &nbsp; [Jay Chou Lyrics Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen_jay_test.py) &nbsp; | &nbsp; 
