from cdcgan import CDCGAN
import sklearn
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


def gen_batch(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i : i+batch_size]


def scaled(images):
    return (images.astype(np.float32) - (255./2)) / (255./2)


def main(args):
    (X, y), (_, _) = tf.keras.datasets.mnist.load_data()
    X = scaled(X)[:, :, :, np.newaxis]
    Y = tf.keras.utils.to_categorical(y)
    print("Data Loaded")
    gan = CDCGAN(args['g_size'], (28, 28), 1, shape_trace=[(7, 7, 64), (14, 14, 32)], n_out=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    plt.figure()
    for i, epoch in enumerate(range(args['n_epoch'])):
        X, Y = sklearn.utils.shuffle(X, Y)
        print("Data Shuffled")
        for step, (images, labels) in enumerate(zip(gen_batch(X, args['batch_size']),
                                                    gen_batch(Y, args['batch_size']))):
            noise = np.random.randn(len(images), args['g_size'])
            sess.run(gan.D_train, {gan.G_in:noise, gan.label:labels, gan.X_in:images, gan.train_flag:True})
            for _ in range(2):
                sess.run(gan.G_train, {gan.G_in:noise, gan.label:labels, gan.train_flag:True})
            
            G_loss, D_loss, D_prob, G_prob, mse = sess.run([gan.G_loss, gan.D_loss, gan.X_prob, gan.G_prob, gan.mse],
                {gan.G_in:noise, gan.label:labels, gan.X_in:images, gan.train_flag:False})
            print("Epoch %d/%d | Step %d/%d" % (epoch+1, args['n_epoch'], step, len(X)//args['batch_size']))
            print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | mse: %.4f" %
                 (G_loss, D_loss, D_prob.mean(), G_prob.mean(), mse))

        if i in range(args['n_epoch']-4, args['n_epoch']):
            target = [0]*10
            target[8] = 1
            img = sess.run(gan.G_out, {gan.G_in:np.random.randn(1, args['g_size']), gan.label:[target],
                                       gan.train_flag:False})[0]
            plt.subplot(2, 2, i+1-(args['n_epoch']-4))
            plt.imshow(np.squeeze(img))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = {
       'n_epoch': 8,
       'batch_size': 64,
       'g_size': 100, 
    }
    main(args)