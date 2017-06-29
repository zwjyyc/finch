![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)

## Contents
* [机器学习](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#机器学习)
      * [线性模型](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#线性模型)
      * [支持向量机](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#支持向量机)
      * [集成](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#集成)
      * [分解](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#分解)
* [Deep Learning](https://github.com/zhedongzheng/finch/blob/master/README.md#deep-learning)
    * [Multilayer Perceptron](https://github.com/zhedongzheng/finch/blob/master/README.md#multilayer-perceptron)
    * [Convolutional Network](https://github.com/zhedongzheng/finch/blob/master/README.md#convolutional-network)
    * [Recurrent Network](https://github.com/zhedongzheng/finch/blob/master/README.md#recurrent-network)
    * [Autoencoder](https://github.com/zhedongzheng/finch/blob/master/README.md#autoencoder)
    * [Highway Network](https://github.com/zhedongzheng/finch/blob/master/README.md#highway-network)
    * [Generative Adversarial Network](https://github.com/zhedongzheng/finch/blob/master/README.md#generative-adversarial-network)
* [Natural Language Processing](https://github.com/zhedongzheng/finch/blob/master/README.md#natural-language-processing)
    * [Preprocessing](https://github.com/zhedongzheng/finch/blob/master/README.md#preprocessing)
    * [Language Model](https://github.com/zhedongzheng/finch/blob/master/README.md#language-model)
    * [Text Classification](https://github.com/zhedongzheng/finch/blob/master/README.md#text-classification)
    * [Text Generation](https://github.com/zhedongzheng/finch/blob/master/README.md#text-generation)
    * [POS Tagging](https://github.com/zhedongzheng/finch/blob/master/README.md#pos-tagging)
    * [Segmentation](https://github.com/zhedongzheng/finch/blob/master/README.md#segmentation)
* [Computer Vision](https://github.com/zhedongzheng/finch/blob/master/README.md#computer-vision)
    * [OpenCV](https://github.com/zhedongzheng/finch/blob/master/README.md#opencv)

## 机器学习
#### 线性模型
* TensorFlow &nbsp; | &nbsp; 线性回归 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 逻辑回归 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic_test.py) &nbsp; | &nbsp;

* Java &nbsp; | &nbsp; 逻辑回归 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;
#### 支持向量机
* TensorFlow &nbsp; | &nbsp; 线性支持向量机 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf_test.py) &nbsp; | &nbsp;

* Java &nbsp; | &nbsp; 线性支持向量机 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp;

* Libsvm &nbsp; | &nbsp; 非线性 支持向量机 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/libsvm_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/libsvm_clf_test.py) &nbsp; | &nbsp;
#### 集成
* NumPy &nbsp; | &nbsp; Bagging 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf_test.py) &nbsp; | &nbsp;

* NumPy &nbsp; | &nbsp; Adaboost 分类器 &nbsp; &nbsp; [伪代码](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.md) &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf_test.py) &nbsp; | &nbsp;

* NumPy &nbsp; | &nbsp; 随机森林 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf_test.py) &nbsp; | &nbsp;
#### 分解
* TensorFlow &nbsp; | &nbsp; 非负矩阵分解 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/decomposition/nmf.py) &nbsp; &nbsp; [MovieLens数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/decomposition/nmf_movielens_test.py) &nbsp; | &nbsp;
## Deep Learning
#### Multilayer Perceptron
* TensorFlow &nbsp; | &nbsp; MLP Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; MLP Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
#### Convolutional Network
* TensorFlow &nbsp; | &nbsp; Conv2D Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Conv2D Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### Recurrent Network
* TensorFlow &nbsp; | &nbsp; LSTM Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM Regressor &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr_plot.py) &nbsp; &nbsp; [Preview](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; GRU Regressor &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_regr_plot.py) &nbsp; | &nbsp;

#### Autoencoder
* TensorFlow &nbsp; | &nbsp; MLP Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; | &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
#### Highway Network
* TensorFlow &nbsp; | &nbsp; MLP Highway Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv1D Highway Classifier &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_hn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_hn_text_clf_imdb_test.py) &nbsp; | &nbsp;

#### Generative Adversarial Network
* TensorFlow &nbsp; | &nbsp; MLP GAN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan_test.py) &nbsp; | &nbsp; MLP Conditional GAN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_cond_gan.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_cond_gan_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv GAN &nbsp; &nbsp; MNIST &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/conv_gan_mnist.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/conv_gan_mnist_test.py) &nbsp; | &nbsp;

## Natural Language Processing
#### Text Preprocessing
* Python &nbsp; | &nbsp; [Text Cleaning](https://github.com/zhedongzheng/finch/blob/master/nlp-models/text-cleaning.ipynb)

* Python &nbsp; | &nbsp; [Word Indexing](https://github.com/zhedongzheng/finch/blob/master/nlp-models/word-indexing.ipynb)

#### Language Model
* Sklearn &nbsp; | &nbsp; LSA &nbsp; &nbsp; [Book Titles Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Tri-Gram &nbsp; &nbsp; [Amazon Review Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/trigram_test.py) &nbsp; | &nbsp;

* Sklearn &nbsp; | &nbsp; TF-IDF &nbsp; &nbsp; [Brown Corpus Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Skip-Gram &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram.py) &nbsp; &nbsp; [Text8 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_text8_test.py) &nbsp; | &nbsp;

#### Text Classification
* Sklearn &nbsp; | &nbsp; TF-IDF + Logistic Regression &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv1D &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; Concat Conv1D &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Conv1D-LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Conv1D &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; Conv1D-LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

#### Text Generation
* TensorFlow &nbsp; | &nbsp; Char-LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen.py) &nbsp; &nbsp; [Anna Karenina Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_anna_test.py) &nbsp; | &nbsp;

#### POS Tagging
* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_seq2seq_clf.py) &nbsp; &nbsp; [CoNLL-2000 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_rnn_test.py) &nbsp; | &nbsp; Bi-directional LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_seq2seq_clf.py) &nbsp; &nbsp; [CoNLL-2000 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_seq_clf.py) &nbsp; &nbsp; [CoNLL-2000 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_tagging_test.py) &nbsp; | &nbsp; Bi-directional LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_seq_clf.py) &nbsp; &nbsp; [CoNLL-2000 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_tagging_test.py) &nbsp; | &nbsp;

#### Segmentation
* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_seq2seq_clf.py) &nbsp; &nbsp; [ICWB2 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_rnn_test.py) &nbsp; | &nbsp; Bi-directional LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_seq2seq_clf.py) &nbsp; &nbsp; [ICWB2 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_seq_clf.py) &nbsp; &nbsp; [ICWB2 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_chseg_test.py) &nbsp; | &nbsp; Bi-directional LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_seq_clf.py) &nbsp; &nbsp; [ICWB2 Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_chseg_test.py) &nbsp; | &nbsp;

## Computer Vision
#### OpenCV
* OP &nbsp; | &nbsp; [Resize](https://github.com/zhedongzheng/finch/blob/master/cv-models/resize.ipynb)

* OP &nbsp; | &nbsp; [Rotations](https://github.com/zhedongzheng/finch/blob/master/cv-models/rotations.ipynb)

* Segmentation &nbsp; | &nbsp; [Contours](https://github.com/zhedongzheng/finch/blob/master/cv-models/contours.ipynb)

* Segmentation &nbsp; | &nbsp; [Sorting Contours](https://github.com/zhedongzheng/finch/blob/master/cv-models/sorting-contours.ipynb)

* Detection &nbsp; | &nbsp; [Face & Eye Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/cv-models/face-eye-detection.ipynb)

* Detection &nbsp; | &nbsp; [Walker & Car Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/cv-models/car-walker-detection.ipynb)
