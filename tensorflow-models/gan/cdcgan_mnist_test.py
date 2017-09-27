from cdcgan import CDCGAN
import sklearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gen_batch(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i : i+batch_size]


def scaled(images):
    return (images.astype(np.float32) - (255./2)) / (255./2)


def main(N_EPOCH=4, BATCH_SIZE=64, G_SIZE=100):
    (X, y), (_, _) = tf.contrib.keras.datasets.mnist.load_data()
    X = scaled(X)[:, :, :, np.newaxis]
    Y = tf.contrib.keras.utils.to_categorical(y)
    print("Data Loaded")
    gan = CDCGAN(G_SIZE, (28, 28), 1, shape_trace=[(7, 7, 64), (14, 14, 32)], n_out=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    plt.figure()
    for i, epoch in enumerate(range(N_EPOCH)):
        X, Y = sklearn.utils.shuffle(X, Y)
        print("Data Shuffled")
        for step, (images, labels) in enumerate(zip(gen_batch(X, BATCH_SIZE), gen_batch(Y, BATCH_SIZE))):
            noise = np.random.randn(len(images), G_SIZE)
            sess.run(gan.D_train, {gan.G_in:noise, gan.label:labels, gan.X_in:images, gan.train_flag:True})
            for _ in range(2):
                sess.run(gan.G_train, {gan.G_in:noise, gan.label:labels, gan.train_flag:True})
            
            G_loss, D_loss, D_prob, G_prob, mse = sess.run([gan.G_loss, gan.D_loss, gan.X_prob, gan.G_prob, gan.mse],
                {gan.G_in:noise, gan.label:labels, gan.X_in:images, gan.train_flag:False})
            print("Epoch %d/%d | Step %d/%d" % (epoch+1, N_EPOCH, step, len(X)//BATCH_SIZE))
            print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | mse: %.4f" %
                 (G_loss, D_loss, D_prob.mean(), G_prob.mean(), mse))

        if i in range(N_EPOCH-4, N_EPOCH):
            target = [0]*10
            target[8] = 1
            img = sess.run(gan.G_out, {gan.G_in:np.random.randn(1, G_SIZE), gan.label:[target],
                                       gan.train_flag:False})[0]
            plt.subplot(2, 2, i+1-(N_EPOCH-4))
            plt.imshow(np.squeeze(img))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()