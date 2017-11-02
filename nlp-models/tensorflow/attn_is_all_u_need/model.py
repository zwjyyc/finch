from config import args
from modules import *

import tensorflow as tf


def _forward_pass(sources, targets, params, reuse=False):
    with tf.variable_scope('forward_pass', reuse=reuse):

        with tf.variable_scope('encoder', reuse=reuse):

            with tf.variable_scope('encoder_embedding'):
                encoded = embed_seq(
                    sources, params['source_vocab_size'], args.hidden_units, zero_pad=True, scale=True)
            
            with tf.variable_scope('positional_encoding'):
                encoded += embed_seq(
                    positional(sources), args.max_len, args.hidden_units, zero_pad=False, scale=False)
            
            with tf.variable_scope('encoder_dropout'):
                encoded = tf.layers.dropout(encoded, args.dropout_rate, training=(not reuse))

            for i in range(args.num_blocks):
                with tf.variable_scope('encoder_attn_%d'%i):
                    encoded = multihead_attn(queries=encoded, keys=encoded, num_units=args.hidden_units,
                        num_heads=args.num_heads, dropout_rate=args.dropout_rate, causality=False, reuse=reuse)
                
                with tf.variable_scope('encoder_feedforward_%d'%i):
                    encoded = feed_forward(encoded, num_units=[4*args.hidden_units, args.hidden_units])
        
        with tf.variable_scope('decoder', reuse=reuse):

            decoder_inputs = _decoder_input_pip(targets, params['start_symbol'])
            with tf.variable_scope('decoder_embedding'):
                decoded = embed_seq(decoder_inputs, params['target_vocab_size'], args.hidden_units,
                    zero_pad=True, scale=True)
            
            with tf.variable_scope('positional_encoding'):
                decoded += embed_seq(positional(decoder_inputs), args.max_len, args.hidden_units,
                    zero_pad=False, scale=False)
            
            with tf.variable_scope('decoder_dropout'):
                decoded = tf.layers.dropout(decoded, args.dropout_rate, training=(not reuse))

            for i in range(args.num_blocks):
                with tf.variable_scope('decoder_self_attn_%d'%i):
                    decoded = multihead_attn(queries=decoded, keys=decoded, num_units=args.hidden_units,
                        num_heads=args.num_heads, dropout_rate=args.dropout_rate, causality=True, reuse=reuse)
                
                with tf.variable_scope('decoder_attn_%d'%i):
                    decoded = multihead_attn(queries=decoded, keys=encoded, num_units=args.hidden_units,
                        num_heads=args.num_heads, dropout_rate=args.dropout_rate, causality=False, reuse=reuse)
                
                with tf.variable_scope('decoder_feedforward_%d'%i):
                    decoded = feed_forward(decoded, num_units=[4*args.hidden_units, args.hidden_units])
        
        with tf.variable_scope('output_layer', reuse=reuse):
             logits = tf.layers.dense(decoded, params['target_vocab_size'])
             ids = tf.argmax(logits, -1)
        return logits, ids


def _model_fn_train(features, mode, params):
    logits, _ = _forward_pass(features['source'], features['target'], params)
    _, _ = _forward_pass(features['source'], features['target'], params, reuse=True)

    targets = features['target']
    masks = tf.to_float(tf.not_equal(targets, 0))

    loss_op = tf.contrib.seq2seq.sequence_loss(
        logits=logits, targets=targets, weights=masks)
    train_op = tf.train.AdamOptimizer().minimize(loss_op,
        global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)


def _model_fn_predict(features, mode, params):
    _, _ = _forward_pass(features['source'], features['target'], params)
    _, ids = _forward_pass(features['source'], features['target'], params, reuse=True)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=ids)


def tf_estimator_model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return _model_fn_train(features, mode, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return _model_fn_predict(features, mode, params)


def _decoder_input_pip(targets, start_symbol):
    start_symbols = tf.cast(tf.fill([tf.shape(targets)[0], 1], start_symbol), tf.int64)
    return tf.concat([start_symbols, targets[:, :-1]], axis=-1)
