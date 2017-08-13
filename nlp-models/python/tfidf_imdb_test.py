import tensorflow as tf
from tfidf_logistic import TfidfLogistic


vocab_size = 20000


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)

    model = TfidfLogistic(vocab_size)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    final_acc = (y_pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
