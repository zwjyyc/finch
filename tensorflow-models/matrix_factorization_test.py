import pandas as pd
import tensorflow as tf
from matrix_factorization import MatrixFactorization


if __name__ == '__main__':
    df = pd.read_csv('./temp/ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2],
                    names=['userid', 'itemid', 'rating'])
    R = pd.pivot_table(df, values='rating', index=['userid'], columns=['itemid'])
    R.fillna(0, inplace=True)
    ans = R[2][1]
    R[2][1] = 0

    sess = tf.Session()
    mf = MatrixFactorization(n_user=R.shape[0], n_item=R.shape[1], n_hidden=100, sess=sess)
    
    mf.sess.run(tf.global_variables_initializer())
    for step in range(5000):
        _, loss = mf.sess.run([mf.train_op, mf.loss], feed_dict={mf.R: R.values})
        if step % 100 == 0:
            print(step, loss)
    print("The prediction of", ans, " is", sess.run(mf.R_pred)[2][1])
