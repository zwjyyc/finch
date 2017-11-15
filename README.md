![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)
* These are the models studied, implemented and tested by me in the past;
* I used lightweight datasets, so even if you don't have GPU, you can still run these scripts;
* If you would like to support, please Star the project, many thanks!

## Guide
1. The following command clones all the files (>300MB);
```
git clone https://github.com/zhedongzheng/finch.git
```

2. Use [contents](https://github.com/zhedongzheng/finch#contents) to find the model and test that may interest you, click on that test
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/addr_0.png" width="600">

3. Find the test file path
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/addr.png" width="600">

4. run on command line
```
cd finch/nlp-models/tensorflow
python rnn_attn_estimator_imdb_test.py
```
## Library
Py3 is perferred, but Py2 should also work in theory (if it doesn't please raise an issue)
* [tensorflow >= 1.4](https://www.tensorflow.org/)
* [scikit-learn](http://scikit-learn.org/)
* [openai-gym](https://github.com/openai/gym)
* [nltk](http://www.nltk.org/)
* [opencv 3](http://opencv.org/)
## Style
* ```Model``` is implemented very early, using ```feed_dict``` (most common but slowest data pipeline);
* I am moving towards ```Estimator```, which is based on [tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator), more efficient;
## Contents
* [NLP](https://github.com/zhedongzheng/finch/blob/master/README.md#nlp)
  * [Word Representation](https://github.com/zhedongzheng/finch/blob/master/README.md#word-representation)
  * [Text Classification](https://github.com/zhedongzheng/finch/blob/master/README.md#text-classification)
  * [Text Generation](https://github.com/zhedongzheng/finch/blob/master/README.md#text-generation)
  * [Text Labelling](https://github.com/zhedongzheng/finch/blob/master/README.md#text-labelling)
  * [Sequence to Sequence](https://github.com/zhedongzheng/finch/blob/master/README.md#sequence-to-sequence)
* [Computer Vision](https://github.com/zhedongzheng/finch/blob/master/README.md#computer-vision)
* [Information Retrieval](https://github.com/zhedongzheng/finch/blob/master/README.md#information-retrieval)
* [Reinforcement Learning](https://github.com/zhedongzheng/finch/blob/master/README.md#reinforcement-learning)
* [Appendix](https://github.com/zhedongzheng/finch/blob/master/README.md#appendix)

## NLP
#### Word Representation
* Python &nbsp; | &nbsp; LSA &nbsp; | &nbsp; [Model for Visualization](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.md) &nbsp; | &nbsp; [Model for Concepts](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept_test.md) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; TF-IDF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Word Embedding - Skip-Gram &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Word Embedding - CBOW &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow_test.py) &nbsp; | &nbsp;
#### Text Classification
* Python &nbsp; | &nbsp; TF-IDF + LR &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_logistic.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; CNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; [Model (Multi-kernel)](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; CNN-LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf_imdb_test.py) &nbsp; | &nbsp; [Estimator](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_estimator.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_estimator_imdb_test.py) &nbsp; &nbsp; [IMDB Config](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_estimator_imdb_config.py) &nbsp; | &nbsp;

#### Text Generation
* Python &nbsp; | &nbsp; 2nd order Markov &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/markov_text_gen.py) &nbsp; &nbsp; [Robert Frost Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/markov_text_gen_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; CNN-Highway-RNN &nbsp; &nbsp; [Paper](https://arxiv.org/abs/1508.06615) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen.py) &nbsp; &nbsp; [PTB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Char-RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen.py) &nbsp; | &nbsp; [English Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_test.py) &nbsp; &nbsp; [Chinese Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_addr_test.py) &nbsp; | &nbsp; [Model (Beam-Search)](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam.py) &nbsp; &nbsp; [English Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; [Varational Recurrent Autoencoder](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/vrae) &nbsp; | &nbsp;

#### Text Labelling
* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_seq2seq_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_rnn_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_rnn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; BiLSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_seq2seq_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; BiLSTM + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_crf_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_crf_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_crf_test.py) &nbsp; | &nbsp;
#### Sequence to Sequence
* TensorFlow &nbsp; | &nbsp; Seq2Seq &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_test.py) &nbsp; | &nbsp; [Estimator](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_estimator.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_estimator_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + BiLSTM Encoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + BiLSTM Encoder + Attention + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; [Attention Is All You Need - Transformer](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/attn_is_all_u_need) &nbsp; | &nbsp; 

## Computer Vision
#### Image Classification
* Python &nbsp; | &nbsp; Bayesian Inference &nbsp; &nbsp; [Pixel Classification](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/cv-models/ucl_compgi14/practicalMixGaussA.ipynb) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; MLP &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; CNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp; [Estimator](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_estimator.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_estimator_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp; [Estimator](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_estimator.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_estimator_test.py) &nbsp; | &nbsp;
#### Image Generation
* Autoencoder
    * TensorFlow &nbsp; | &nbsp; Stacked Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp;

    * TensorFlow &nbsp; | &nbsp; Denoising Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/denoising_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/denoising_ae_mnist_test.py) &nbsp; | &nbsp;

    * TensorFlow &nbsp; | &nbsp; Sparse Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/sparse_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/sparse_ae_mnist_test.py) &nbsp; | &nbsp;

    * TensorFlow &nbsp; | &nbsp; Variational Autoencoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/variational_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/variational_ae_mnist_test.py) &nbsp; | &nbsp;

    * TensorFlow &nbsp; | &nbsp; Conv2D Autoencoder (weights-tied) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; &nbsp; [CIFAR10 Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;

* Generative Adversarial Network
    * TensorFlow &nbsp; | &nbsp; DCGAN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan_mnist_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan_mnist_test.md) &nbsp; | &nbsp; [Conditional Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/cdcgan.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/cdcgan_mnist_test.py) &nbsp; | &nbsp;
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

## Information Retrieval
* Python &nbsp; | &nbsp; Apriori &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/apriori.py) &nbsp; &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/apriori_movielens_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Collborative Filtering &nbsp; | &nbsp; MovieLens &nbsp; &nbsp; [User-based Model](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/ncf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/ncf_movielens_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Matrix Factorization &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/ir-models/tensorflow/nmf.py) &nbsp; &nbsp; [MovieLens Test](https://github.com/zhedongzheng/finch/blob/master/ir-models/tensorflow/nmf_movielens_test.py) &nbsp; | &nbsp;

## Reinforcement Learning
* Python &nbsp; | &nbsp; Q-Learning &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/rl-models/python/q.py) &nbsp; &nbsp; [CartPole Test](https://github.com/zhedongzheng/finch/blob/master/rl-models/python/q_cartpole_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Sarsa &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/rl-models/python/sarsa.py) &nbsp; &nbsp; [CartPole Test](https://github.com/zhedongzheng/finch/blob/master/rl-models/python/sarsa_cartpole_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Policy Gradient &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/rl-models/tensorflow/pg.py) &nbsp; &nbsp; [CartPole Test](https://github.com/zhedongzheng/finch/blob/master/rl-models/tensorflow/pg_cartpole_test.py) &nbsp; | &nbsp;

## Appendix
#### Shallow Structure Models
* Java &nbsp; | &nbsp; Logistic Regression &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;

* Java &nbsp; | &nbsp; Support Vector Machine &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Linear Regression &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Logistic Regression &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Support Vector Machine &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; K Nearest Neighbors &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/knn.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/knn_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; K-Means &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/kmeans.py) &nbsp; &nbsp; [MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/kmeans_mnist_test.py) &nbsp; | &nbsp;
#### Ensemble
* Python &nbsp; | &nbsp; Bagging &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Adaboost &nbsp; &nbsp; [Pseudocode](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.md) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; Random Forest &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Random Forest &nbsp; &nbsp; [Estimator & MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/forest_mnist.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Gradient Boosting Trees &nbsp; &nbsp; [Estimator & MNIST Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/shallow/gbt_mnist.py) &nbsp; | &nbsp;
#### Time Series
* TensorFlow &nbsp; | &nbsp; RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr_plot.py) &nbsp; &nbsp; [Preview](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp;
