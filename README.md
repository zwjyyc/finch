![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)

This repository contains a wide range of my API models and tests written on machine learning topics

The models are designed to be:
* easy to plug in

  ```python
  from rnn_clf import RNNClassifier
  ......
  clf = RNNClassifier(n_in=28, n_step=28, n_out=10, stateful=True)
  log = clf.fit(X_train, Y_train, val_data=(X_test,Y_test))
  pred = clf.predict(X_test)
  ```
* easy to read with a overview provided in method build_graph() for each model

  ```python
  def build_graph(self):
      self.add_input_layer()
      self.add_word_embedding()
      self.add_conv1d('conv', filter_shape=[self.kernel_size, self.embedding_dims, self.n_filters])
      self.add_pooling(self.pool_size)
      self.add_lstm_cells()
      self.add_dynamic_rnn()
      self.add_output_layer()   
      self.add_backward_path()
  # end method build_graph
  ```

work in process ......

## Contents
* [Machine Learning](https://github.com/zhedongzheng/finch/blob/master/README.md#machine-learning)
  * [Linear Model](https://github.com/zhedongzheng/finch/blob/master/README.md#linear-model)
  * [Support Vector Machine](https://github.com/zhedongzheng/finch/blob/master/README.md#support-vector-machine)
  * [Ensemble](https://github.com/zhedongzheng/finch/blob/master/README.md#ensemble)
  * [Decomposition](https://github.com/zhedongzheng/finch/blob/master/README.md#decomposition)
* [Deep Learning](https://github.com/zhedongzheng/finch/blob/master/README.md#deep-learning)
  * [Multilayer Perceptron](https://github.com/zhedongzheng/finch/blob/master/README.md#multilayer-perceptron)
  * [Convolutional Network](https://github.com/zhedongzheng/finch/blob/master/README.md#convolutional-network)
  * [Recurrent Network](https://github.com/zhedongzheng/finch/blob/master/README.md#recurrent-network)
  * [Recurrent Convolutional Network](https://github.com/zhedongzheng/finch/blob/master/README.md#recurrent-convolutional-network)
  * [Autoencoder](https://github.com/zhedongzheng/finch/blob/master/README.md#autoencoder)
  * [Highway Network](https://github.com/zhedongzheng/finch/blob/master/README.md#highway-network)
  * [Generative Adversarial Network](https://github.com/zhedongzheng/finch/blob/master/README.md#generative-adversarial-network)
* [Computer Vision](https://github.com/zhedongzheng/finch/blob/master/README.md#computer-vision)
  * [Basic Operations](https://github.com/zhedongzheng/finch/blob/master/README.md#basic-operations)
  * [Image Segmentation](https://github.com/zhedongzheng/finch/blob/master/README.md#image-segmentation)
  * [Detection](https://github.com/zhedongzheng/finch/blob/master/README.md#detection)
  * [Deep CV](https://github.com/zhedongzheng/finch/blob/master/README.md#deep-cv)
* [Natural Language Processing](https://github.com/zhedongzheng/finch/blob/master/README.md#natural-language-processing)
  * [Preprocessing](https://github.com/zhedongzheng/finch/blob/master/README.md#preprocessing)
  * [Deep NLP](https://github.com/zhedongzheng/finch/blob/master/README.md#deep-nlp)

## Machine Learning
#### Linear Model
* TensorFlow &nbsp; | &nbsp; Linear Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Logistic Regression &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;
#### Support Vector Machine
* TensorFlow &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf_test.py) &nbsp; | &nbsp;
* Java &nbsp; | &nbsp; Linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp;
* Libsvm &nbsp; | &nbsp; Non-linear SVM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/libsvm_clf.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/libsvm_clf_test.py) &nbsp; | &nbsp;
#### Ensemble
* Python &nbsp; | &nbsp; Bagging Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf_test.py) &nbsp; | &nbsp;
#### Decomposition
* TensorFlow &nbsp; | &nbsp; Non-negative Matrix Factorization &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/decomposition/nmf.py) &nbsp; | &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/decomposition/nmf_movielens_test.py) &nbsp; | &nbsp;
## Deep Learning
#### Multilayer Perceptron
* TensorFlow &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; MLP Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp;
* PyTorch &nbsp; | &nbsp; MLP Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
#### Convolutional Network
* TensorFlow &nbsp; | &nbsp; Conv1D Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_1d_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
* PyTorch &nbsp; | &nbsp; Conv2D Image Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Network
* TensorFlow &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Time Series Regressor &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr_plot.py) &nbsp; &nbsp; [Preview](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
* TensorFlow &nbsp; | &nbsp; LSTM Char Language Model &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen.py) &nbsp; | &nbsp; [Anna Karenina Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_text_gen_anna_test.py) &nbsp; | &nbsp;
* PyTorch &nbsp; | &nbsp; LSTM Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Convolutional Network
* TensorFlow &nbsp; | &nbsp; Conv1D-LSTM Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn_rnn/conv_rnn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn_rnn/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;
#### Autoencoder
* TensorFlow &nbsp; | &nbsp; MLP Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; | &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
#### Highway Network
* TensorFlow &nbsp; | &nbsp; MLP Highway Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf_mnist_test.py) &nbsp; | &nbsp;
* TensorFlow &nbsp; | &nbsp; Conv1D Highway Text Classifier &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/conv_1d_hn_text_clf.py) &nbsp; | &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/conv_1d_hn_text_clf_imdb_test.py) &nbsp; | &nbsp; 
#### Generative Adversarial Network
* TensorFlow &nbsp; | &nbsp; MLP GAN &nbsp; | &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan.py) &nbsp; | &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan_test.py) &nbsp; | &nbsp;
## Computer Vision
#### Basic Operations
  * OpenCV &nbsp; | &nbsp; [Resize](https://github.com/zhedongzheng/finch/blob/master/cv-models/resize.ipynb)
  * OpenCV &nbsp; | &nbsp; [Rotations](https://github.com/zhedongzheng/finch/blob/master/cv-models/rotations.ipynb)
#### Image Segmentation
  * OpenCV &nbsp; | &nbsp; [Contours](https://github.com/zhedongzheng/finch/blob/master/cv-models/contours.ipynb)
  * OpenCV &nbsp; | &nbsp; [Sorting Contours](https://github.com/zhedongzheng/finch/blob/master/cv-models/sorting-contours.ipynb)
#### Detection
  * OpenCV &nbsp; | &nbsp; [Face & Eye Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/cv-models/face-eye-detection.ipynb)
  * OpenCV &nbsp; | &nbsp; [Walker & Car Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/cv-models/car-walker-detection.ipynb)
## Natural Language Processing
#### Preprocessing
* Python &nbsp; | &nbsp; [Text Cleaning](https://github.com/zhedongzheng/finch/blob/master/nlp-models/text-cleaning.ipynb)
* Python &nbsp; | &nbsp; [Word Indexing](https://github.com/zhedongzheng/finch/blob/master/nlp-models/word-indexing.ipynb)
