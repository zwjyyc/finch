import tensorflow as tf


ones = tf.ones((4, 4))
tril = tf.contrib.linalg.LinearOperatorTriL(ones).to_dense()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(ones))
print()
print(sess.run(tril))
