import numpy as np
from rnn_regr import RNNRegressor
import tensorflow as tf


BATCH_START = 0     
TIME_STEPS = 20     
BATCH_SIZE = 50  


def get_data_batch(return_x=False):
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS*BATCH_SIZE
    if return_x:
        [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]
    else:
        return (seq[:, :, np.newaxis], res[:, :, np.newaxis])


if __name__ == '__main__':
    data = [get_data_batch() for _ in range(2000)]
    train_data = data[:1500]
    test_data = data[1500:]
    sess = tf.Session()
    model = RNNRegressor(sess,
                         n_step = TIME_STEPS,
                         n_in = 1,
                         n_out = 1,
                         cell_size = 16)

    model.sess.run(tf.global_variables_initializer())
    train_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})
    for train_idx, train_sample in enumerate(train_data):
        seq, res = train_sample
        _, train_loss, train_state = model.sess.run([model.train_op, model.loss, model.final_state],
                                                     feed_dict={model.X:seq, model.Y:res,
                                                                model.init_state:train_state,
                                                                model.batch_size:BATCH_SIZE})
        if test_data is None:
            if train_idx % 20 == 0:
                print('train loss: %.4f' % (train_loss))
        else:
            test_loss_list = []
            test_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})
            for test_idx, test_sample in enumerate(test_data):
                seq_test, res_test = test_sample
                test_loss, test_state = model.sess.run([model.loss, model.final_state],
                                                        feed_dict={model.X:seq_test, model.Y:res_test,
                                                                   model.init_state:test_state,
                                                                   model.batch_size:BATCH_SIZE})
                test_loss_list.append(test_loss)
            if train_idx % 20 == 0:
                print('train loss: %.4f |' % (train_loss),
                      'test loss: %.4f' % (sum(test_loss_list)/len(test_loss_list)))
