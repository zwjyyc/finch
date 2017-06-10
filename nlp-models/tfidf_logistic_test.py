import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression


vocab_size = 20000


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)

    indexed = np.concatenate([X_train, X_test])
    DT = np.zeros((len(indexed), vocab_size))
    i = 0
    for indices in indexed:
        for idx in indices:
            DT[i, idx] += 1
        i += 1
    print("Document-Term matrix built ...")

    model = TfidfTransformer()
    DT = model.fit_transform(DT).toarray()
    print("TF-IDF transform completed ...")
    
    X_train = DT[:len(X_train)]
    X_test = DT[len(X_train):]

    model = LogisticRegression()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    
    final_acc = (y_pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
