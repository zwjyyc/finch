from config import args

import numpy as np
import tensorflow as tf


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.Variable(tf.ones(params_shape))
    beta = tf.Variable(tf.zeros(params_shape))
    
    outputs = gamma * normalized + beta
    return outputs


def embed_seq(inputs, vocab_size, embed_dim, zero_pad=False, scale=False):
    lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, embed_dim],
        initializer=tf.glorot_uniform_initializer())
    if zero_pad:
        lookup_table = tf.concat((tf.zeros([1, embed_dim]), lookup_table[1:, :]), axis=0)
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    if scale:
        outputs = outputs * (embed_dim ** 0.5) 
    return outputs


def multihead_attn(queries, keys, num_units=None, num_heads=8,
        dropout_rate=args.dropout_rate, causality=False, reuse=False):
    """
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q]
      keys: A 3d tensor with shape of [N, T_k, C_k]
    """
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]
    T_q = queries.get_shape().as_list()[1]
    T_k = keys.get_shape().as_list()[1]

    Q = tf.layers.dense(queries, num_units, tf.nn.relu)                            # (N, T_q, C)
    K = tf.layers.dense(keys, num_units, tf.nn.relu)                               # (N, T_k, C)
    V = tf.layers.dense(keys, num_units, tf.nn.relu)                               # (N, T_k, C)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         # (h*N, T_q, C/h) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h)

    outputs = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                             # (h*N, T_q, T_k)
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)                      # scale

    # Key Masking
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))                      # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])                                 # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])                 # (h*N, T_q, T_k)
    paddings = tf.ones_like(outputs) * (-2**32)                                    # exp(-large) -> 0
    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)                  # (h*N, T_q, T_k)

    if causality:
        lower_tri = tf.ones([T_q, T_k])                                            # (T_q, T_k)
        lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()     # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)                  # (h*N, T_q, T_k)
    
    outputs = tf.nn.softmax(outputs)                                               # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))                 # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])                             # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])            # (h*N, T_q, T_k)
    outputs *= query_masks                                                         # (h*N, T_q, T_k)

    outputs = tf.layers.dropout(outputs, dropout_rate, training=(not reuse))       # (h*N, T_q, T_k)

    # Weighted sum
    outputs = tf.matmul(outputs, V_)                                               # (h*N, T_q, C/h)
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)              # (N, T_q, C)
    # Residual connection
    outputs += queries                                                             # (N, T_q, C)   
    # Normalize
    outputs = layer_norm(outputs)                                                 # (N, T_q, C)
    return outputs


def feed_forward(inputs, num_units=[2048, 512]):
    # Inner layer
    outputs = tf.layers.conv1d(inputs, num_units[0], kernel_size=1, activation=tf.nn.relu)
    # Readout layer
    outputs = tf.layers.conv1d(inputs, num_units[1], kernel_size=1, activation=tf.nn.relu)
    # Residual connection
    outputs += inputs
    # Normalize
    outputs = layer_norm(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    C = inputs.get_shape().as_list()[-1] # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / C)


def positional(inputs):
    outputs = tf.range(tf.shape(inputs)[1])                # (T_q)
    outputs = tf.expand_dims(outputs, 0)                   # (1, T_q)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])   # (N, T_q)
    return outputs