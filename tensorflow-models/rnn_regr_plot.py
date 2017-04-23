import numpy as np
from rnn_regr import RNNRegressor
import tensorflow as tf
import matplotlib.pyplot as plt


BATCH_START = 0     
TIME_STEPS = 20     
BATCH_SIZE = 50  


def get_data_batch(return_x=False):
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS*BATCH_SIZE
    # returned seq, res: shape (batch, step, input)
    if return_x:
        return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]
    else:
        return (seq[:, :, np.newaxis], res[:, :, np.newaxis])


if __name__ == '__main__':
    data = [get_data_batch(return_x=True) for _ in range(2000)]
    train_data = data[:1000]
    test_data = data[1000:]
    sess = tf.Session()
    model = RNNRegressor(n_in=1, n_step=TIME_STEPS, rnn_size=16, n_out=1, sess=sess)

    model.sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.show()

    train_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})
    test_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})
    for train_idx, train_sample in enumerate(train_data):
        seq, res, xs = train_sample
        _, train_loss, train_state = model.sess.run( [model.train_op, model.loss, model.final_state],
            feed_dict={model.X:seq, model.y:res, model.init_state:train_state, model.batch_size:BATCH_SIZE})

        test_sample = test_data[train_idx]
        seq_test, res_test, xs_test = test_sample
        test_loss, test_state, test_pred = model.sess.run([model.loss, model.final_state, model.time_seq_out],
            feed_dict={model.X:seq_test, model.y:res_test, model.init_state:test_state,
                       model.batch_size:BATCH_SIZE})
        
        # update plotting
        plt.plot(xs.ravel(), res_test.ravel(), 'r', xs.ravel(), test_pred.ravel(), 'b--')
        plt.ylim((-1.2, 1.2))
        plt.xlim((xs.ravel()[0], xs.ravel()[-1]))
        plt.draw()
        plt.pause(0.3)

        if train_idx % 20 == 0:
            print('train loss: %.4f | test loss: %.4f' % (train_loss, test_loss)) 
