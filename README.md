* [TensorFlow >= 1.4](https://www.tensorflow.org/)
* [scikit-learn](http://scikit-learn.org/)
* [NLTK](http://www.nltk.org/)
---
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
---
#### Word Representation
* Python &nbsp; | &nbsp; TF-IDF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_brown_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Skip-Gram &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_skipgram_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; CBOW &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/word2vec_cbow_test.py) &nbsp; | &nbsp;

#### Topic Modelling
* Python &nbsp; | &nbsp; LSA &nbsp; | &nbsp; [Model for Visualization](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_test.md) &nbsp; | &nbsp; [Model for Concepts](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept.py) &nbsp; &nbsp; [Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/lsa_concept_test.md) &nbsp; | &nbsp;

#### Text Classification
* Python &nbsp; | &nbsp; TF-IDF + LR &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_logistic.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/python/tfidf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; CNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_1d_text_clf_imdb_test.py) &nbsp; | &nbsp; [Model (Multi-kernel)](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.py) &nbsp; &nbsp; [Result](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/concat_conv_1d_text_clf_imdb_test.md) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; CNN-LSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/conv_rnn_text_clf_imdb_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; LSTM + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_text_clf_imdb_test.py) &nbsp; | &nbsp; [Estimator](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_estimator.py) &nbsp; &nbsp; [IMDB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_estimator_imdb_test.py) &nbsp; &nbsp; [IMDB Config](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_attn_estimator_imdb_config.py) &nbsp; | &nbsp;

#### Sequence Labelling
* TensorFlow &nbsp; | &nbsp; BiLSTM &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_seq2seq_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_test.py) &nbsp; | &nbsp;

    * TensorFlow &nbsp; | &nbsp; BiLSTM + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/birnn_crf_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/pos_birnn_crf_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/chseg_birnn_crf_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Self-Attention &nbsp; &nbsp; [Modules](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_pos_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Self-Attention + CRF &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf.py) &nbsp; | &nbsp; [POS Tagging Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf_pos_test.py) &nbsp; | &nbsp; [Chinese Segmentation Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/multihead_attn_clf_chseg_test.py) &nbsp; | &nbsp;

#### Text Generation
* TensorFlow &nbsp; | &nbsp; Char-RNN &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen.py) &nbsp; | &nbsp; [English Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_test.py) &nbsp; &nbsp; [Chinese Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_gen_addr_test.py) &nbsp; | &nbsp;
   * TensorFlow &nbsp; | &nbsp; Char-RNN + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam.py) &nbsp; &nbsp; [English Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/char_rnn_beam_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; Character-Aware CNN-RNN Language Model &nbsp; &nbsp; [Paper](https://arxiv.org/abs/1508.06615) &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen.py) &nbsp; &nbsp; [PTB Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/cnn_rnn_text_gen_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; [Variational Autoencoder (VAE)](https://github.com/zhedongzheng/vae-nlp) &nbsp; | &nbsp; 

#### Sequence to Sequence
* TensorFlow &nbsp; | &nbsp; Seq2Seq &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_test.py) &nbsp; | &nbsp; [Estimator](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_estimator.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_estimator_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + Attention &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + BiLSTM Encoder &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_birnn_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_beam_test.py) &nbsp; | &nbsp;

   * TensorFlow &nbsp; | &nbsp; Seq2Seq + BiLSTM Encoder + Attention + Beam-Search &nbsp; &nbsp; [Model](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate.py) &nbsp; &nbsp; [Sorting Test](https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_ultimate_test.py) &nbsp; | &nbsp;

* TensorFlow &nbsp; | &nbsp; [Transformer (Attention Is All You Need)](https://github.com/zhedongzheng/finch/tree/master/nlp-models/tensorflow/attn_is_all_u_need) &nbsp; | &nbsp; 
