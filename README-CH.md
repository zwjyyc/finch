![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)

欢迎写邮件到我的邮箱，任何问题都可以，看到就回复。

## 安装
首先下载本目录，可以在命令行用下面的 git 命令，也可以在网页上用右上角的 download zip
```
git clone https://github.com/zhedongzheng/finch.git
```
然后可以直接运行任何测试脚本
```
python xxxx_test.py
```
主要依赖于下面三个库：
* [TensorFlow 1.2](https://www.tensorflow.org/)
* [PyTorch 0.12](http://pytorch.org/)
* [scikit-learn](http://scikit-learn.org/)

## 编码风格
GitHub 上的大多数深度学习项目通常会有若干个脚本比如 ```train.py```, ```model.py```, ```main.py``` 等等。我个人倾向于把功能融合成一个带有 ```fit()``` 和 ```predict()``` 接口的模型（scikit-learn API 风格），然后对于不同的数据集编写不同的测试文件。下面所有代码都遵从这种风格。

## 目录
* [机器学习](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#机器学习)
    * [线性模型](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#线性模型)
    * [支持向量机](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#支持向量机)
    * [集成](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#集成)
* [深度学习](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#深度学习)
    * [多层感知](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#多层感知)
    * [卷积网络](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#卷积网络)
    * [循环网络](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#循环网络)
    * [自动解码](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#自动解码)
    * [高速公路网络](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#高速公路网络)
    * [对抗生成网络](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#对抗生成网络)
* [自然语言处理](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#自然语言处理)
    * [预处理](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#预处理)
    * [语言模型](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#语言模型)
    * [文本分类](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#文本分类)
    * [文本生成](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#文本生成)
    * [序列标注](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#序列标注)
    * [序列到序列](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#序列到序列)
* [信息检索](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#信息检索)
    * [推荐系统](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#推荐系统)
* [计算机视觉](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#计算机视觉)
    * [OpenCV](https://github.com/zhedongzheng/finch/blob/master/README-CH.md#opencv)

## 机器学习
#### 线性模型
* TensorFlow &nbsp; | &nbsp; 线性回归 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/linear_regr_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 逻辑回归 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/linear_model/logistic_test.py) &nbsp; | &nbsp;

* Java &nbsp; | &nbsp; 逻辑回归 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegression.java) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/java-models/LogisticRegressionTest.java) &nbsp; | &nbsp;
#### 支持向量机
* TensorFlow &nbsp; | &nbsp; 线性 支持向量机 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/svm/svm_linear_clf_test.py) &nbsp; | &nbsp;

* Java &nbsp; | &nbsp; 线性 支持向量机 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVM.java) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/java-models/LinearSVMTest.java) &nbsp; | &nbsp;

* Libsvm &nbsp; | &nbsp; 非线性 支持向量机 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/libsvm_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/libsvm_clf_test.py) &nbsp; | &nbsp;
#### 集成
* NumPy &nbsp; | &nbsp; Bagging 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/bagging_clf_test.py) &nbsp; | &nbsp;

* NumPy &nbsp; | &nbsp; Adaboost 分类器 &nbsp; &nbsp; [伪代码](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.md) &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/adaboost_clf_test.py) &nbsp; | &nbsp;

* NumPy &nbsp; | &nbsp; 随机森林 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/classic-models/random_forest_clf_test.py) &nbsp; | &nbsp;

## 深度学习
#### 多层感知
* TensorFlow &nbsp; | &nbsp; 多层感知 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; 多层感知 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10数据集测试](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/mlp/mlp_clf_cifar10_test.py) &nbsp; | &nbsp; 
#### 卷积网络
* TensorFlow &nbsp; | &nbsp; 二维卷积 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/cnn/conv_2d_clf_cifar10_keras_idg_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; 二维卷积 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10数据集测试](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/cnn/cnn_clf_cifar10_test.py) &nbsp; | &nbsp;
#### 循环网络
* TensorFlow &nbsp; | &nbsp; LSTM 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM 回归器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/rnn/rnn_regr_plot.py) &nbsp; &nbsp; [预览](https://github.com/zhedongzheng/finch/blob/master/assets/rnn_regr_plot.gif) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; LSTM 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_mnist_test.py) &nbsp; &nbsp; [CIFAR10数据集测试](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_clf_cifar10_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; GRU 回归器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_regr.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/pytorch-models/rnn/rnn_regr_plot.py) &nbsp; | &nbsp;

#### 自动解码
* TensorFlow &nbsp; | &nbsp; 多层感知 自动解码 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae.py) &nbsp; | &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/mlp_ae_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 二维卷积 自动解码 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_mnist_test.py) &nbsp; &nbsp; [CIFAR10数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/autoencoder/conv_ae_cifar10_test.py) &nbsp; | &nbsp;
#### 高速公路网络
* TensorFlow &nbsp; | &nbsp; 基于高速公路的 多层感知 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf.py) &nbsp; &nbsp; [MNIST数据集测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/highway/mlp_hn_clf_mnist_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 基于高速公路的 一维卷积 分类器 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_hn_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_hn_text_clf_imdb_test.py) &nbsp; | &nbsp;

#### 对抗生成网络
* TensorFlow &nbsp; | &nbsp; 基于多层感知的 对抗生成网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_gan_test.py) &nbsp; | &nbsp; 有条件限制的 对抗生成网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_cond_gan.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/mlp_cond_gan_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 基于卷积网络的 对抗生成网络 &nbsp; &nbsp; MNIST数据集 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan_mnist_test.py) &nbsp; &nbsp; [结果](https://github.com/zhedongzheng/finch/blob/master/tensorflow-models/gan/dcgan_mnist_test.md) &nbsp; | &nbsp;

## 自然语言处理
#### 预处理
* Python &nbsp; | &nbsp; [文本格式化](https://github.com/zhedongzheng/finch/blob/master/nlp-models/text-cleaning.ipynb)

* Python &nbsp; | &nbsp; [词语索引](https://github.com/zhedongzheng/finch/blob/master/nlp-models/word-indexing.ipynb)

#### 语言模型
* Sklearn &nbsp; | &nbsp; 隐含语义分析 &nbsp; &nbsp; [Wiki资料](https://en.wikipedia.org/wiki/Latent_semantic_analysis) &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.py) &nbsp; | &nbsp;

* Python &nbsp; | &nbsp; 三元词袋 &nbsp; &nbsp; [Wiki资料](https://en.wikipedia.org/wiki/Trigram) &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/trigram.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/trigram_test.py) &nbsp; | &nbsp;

* Sklearn &nbsp; | &nbsp; TF-IDF &nbsp; &nbsp; [Brown文集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.py) &nbsp; &nbsp; [结果](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 词向量 Skip-Gram &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram.py) &nbsp; &nbsp; [Text8文集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_text8_test.py) &nbsp; | &nbsp;

#### 文本分类
* Sklearn &nbsp; | &nbsp; TF-IDF + 逻辑回归 &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 一维卷积 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; 多通道 一维卷积 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.py) &nbsp; &nbsp; [结果](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 双层循环网络 + 注意力机制 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_attn_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_attn_text_clf_imdb_test.py) &nbsp; &nbsp; [结果](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_attn_text_clf_imdb_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 一维卷积+循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; 一维卷积 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; 循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; 一维卷积+循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_rnn_text_clf.py) &nbsp; &nbsp; [IMDB数据集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/cnn_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

#### 文本生成
* Python &nbsp; | &nbsp; 二阶马尔可夫 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/markov_text_gen.py) &nbsp; &nbsp; [Robert Frost 文集测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/markov_text_gen_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 字符循环网络 &nbsp; &nbsp; [Karpathy大神当时的文章](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 卷积网络+循环网络 &nbsp; &nbsp; [原论文](https://arxiv.org/abs/1508.06615) &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen.py) &nbsp; &nbsp; [测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen_test.py) &nbsp; | &nbsp;

#### 序列标注
* TensorFlow &nbsp; | &nbsp; 循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_seq2seq_clf.py) &nbsp; | &nbsp; [词性标记测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_rnn_test.py) &nbsp; | &nbsp; [中文分词测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_rnn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 双向循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_seq2seq_clf.py) &nbsp; | &nbsp; [词性标记测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_test.py) &nbsp; | &nbsp; [中文分词测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 双向循环网络+条件随机场 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_crf_clf.py)
    * [词性标记测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_crf_test.py) &nbsp; &nbsp; [结果](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_crf_test.md)
    
    * [中文分词测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_crf_test.py) &nbsp; &nbsp; [结果](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_crf_test.md) 

* PyTorch &nbsp; | &nbsp; 循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_seq_clf.py) &nbsp; | &nbsp; [词性标记测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_tagging_test.py) &nbsp; | &nbsp; [中文分词测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/rnn_chseg_test.py) &nbsp; | &nbsp;

* PyTorch &nbsp; | &nbsp; 双向循环网络 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_seq_clf.py) &nbsp; | &nbsp; [词性标记测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_tagging_test.py) &nbsp; | &nbsp; [中文分词测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/pytorch/birnn_chseg_test.py) &nbsp; | &nbsp;

#### 序列到序列
* TensorFlow &nbsp; | &nbsp; 动态 Seq2Seq &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq.py)

    * [排序测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_test.py)
    
* TensorFlow &nbsp; | &nbsp; 动态 Seq2Seq + 双向编码 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn.py)

    * [排序测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn_test.py)
    
* TensorFlow &nbsp; | &nbsp; 动态 Seq2Seq + 注意力机制 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn.py) &nbsp; &nbsp;

    * [排序测试](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn_test.py) &nbsp; &nbsp; [结果](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn_test.md)

## 信息检索
#### 推荐系统
* Python &nbsp; | &nbsp; 基于邻近空间的协同过滤 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/ncf.py) &nbsp; &nbsp; [MovieLens测试](https://github.com/zhedongzheng/finch/blob/master/ir-models/python/ncf_movielens_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; 矩阵分解 &nbsp; &nbsp; [模型](https://github.com/zhedongzheng/finch/blob/master/ir-models/tensorflow/nmf.py) &nbsp; &nbsp; [MovieLens测试](https://github.com/zhedongzheng/finch/blob/master/ir-models/tensorflow/nmf_movielens_test.py) &nbsp; | &nbsp;

## 计算机视觉
#### OpenCV
* 基本操作 &nbsp; | &nbsp; [调整大小](https://github.com/zhedongzheng/finch/blob/master/cv-models/resize.ipynb)

* 基本操作 &nbsp; | &nbsp; [旋转](https://github.com/zhedongzheng/finch/blob/master/cv-models/rotations.ipynb)

* 分割 &nbsp; | &nbsp; [轮廓](https://github.com/zhedongzheng/finch/blob/master/cv-models/contours.ipynb)

* 分割 &nbsp; | &nbsp; [轮廓排序](https://github.com/zhedongzheng/finch/blob/master/cv-models/sorting-contours.ipynb)

* 探测 &nbsp; | &nbsp; [Face & Eye Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/cv-models/face-eye-detection.ipynb)

* 探测 &nbsp; | &nbsp; [Walker & Car Detection Using Cascade Classifier](https://github.com/zhedongzheng/finch/blob/master/cv-models/car-walker-detection.ipynb)
