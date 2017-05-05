from keras.utils.np_utils import to_categorical as to_one_hot
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from hn_conv_clf import HighwayConvClassifier
import numpy as np
import tensorflow as tf


batch_size = 128
n_epoch = 20


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    sess = tf.Session()
    model = HighwayConvClassifier(sess, (32,32), 3, 10)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    global_step = 0
    model.sess.run(tf.global_variables_initializer()) # initialize all variables
    for epoch in range(n_epoch):

        local_step = 1
        for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size):
            if local_step > int(len(X_train)/batch_size):
                break
            lr = model.decrease_lr(True, global_step, n_epoch, len(X_train), batch_size) 
            _, loss, acc = model.sess.run([model.train_op, model.loss, model.acc],
                                           feed_dict={model.X:X_batch, model.Y:Y_batch,
                                                      model.lr:lr, model.keep_prob:0.5})
            local_step += 1
            global_step += 1
            if local_step % 50 == 0:
                print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, n_epoch, local_step, int(len(X_train)/batch_size), loss, acc, lr))

        val_loss_list, val_acc_list = [], []
        for X_test_batch, Y_test_batch in zip(model.gen_batch(X_test, batch_size),
                                              model.gen_batch(Y_test, batch_size)):
            v_loss, v_acc = model.sess.run([model.loss, model.acc],
                                            feed_dict={model.X:X_test_batch, model.Y:Y_test_batch,
                                                       model.keep_prob:1.0})
            val_loss_list.append(v_loss)
            val_acc_list.append(v_acc)
        val_loss, val_acc = model.list_avg(val_loss_list), model.list_avg(val_acc_list)

        print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                "lr: %.4f" % (lr) )
