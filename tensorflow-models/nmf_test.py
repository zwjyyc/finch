import pandas as pd
import tensorflow as tf
import math
from nmf import NMF


if __name__ == '__main__':
    df = pd.read_csv('./temp/ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2],
                    names=['userid', 'itemid', 'rating'])
    R = pd.pivot_table(df, values='rating', index=['userid'], columns=['itemid'])
    R.fillna(0, inplace=True)

    ans1 = R[2][1]
    R[2][1] = 0
    ans2 = R[200][940]
    R[200][940] = 0
    ans3 = R[900][931]
    R[900][931] = 0

    sess = tf.Session()
    nmf = NMF(n_user=R.shape[0], n_item=R.shape[1], n_hidden=100, sess=sess)
    
    nmf.sess.run(tf.global_variables_initializer())
    for step in range(5000):
        lr = 0.001
        _, loss = nmf.sess.run([nmf.train_op,nmf.loss], feed_dict={nmf.R:R.values,nmf.lr:lr})
        if step % 100 == 0:
            print(step, loss)
    print("The prediction of", ans1, " is", sess.run(nmf.R_pred)[2][1])
    print("The prediction of", ans2, " is", sess.run(nmf.R_pred)[200][940])
    print("The prediction of", ans3, " is", sess.run(nmf.R_pred)[900][931])
