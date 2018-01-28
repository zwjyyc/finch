from data import DataLoader
from model import Model
from config import args
from tqdm import tqdm

import json
import time
import sklearn
import tensorflow as tf
import numpy as np
import pandas as pd


def main():
    print(json.dumps(args.__dict__, indent=4))
    train()


def train():
    dl = DataLoader()
    t0 = time.time()
    model = Model(dl.params)
    print("%.2f secs ==> TF Graph"%(time.time()-t0))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_batch = len(dl.data['train']['Y']) // args.batch_size
    for epoch in range(1, args.num_epochs+1):
        if epoch > 1:
            dl.data['train']['X'], dl.data['train']['Y'] = sklearn.utils.shuffle(
                dl.data['train']['X'], dl.data['train']['Y'])
            print("<Data Shuffled>")
        for i, (x, y) in enumerate(dl.next_train_batch()):
            loss, lr = model.train_batch(sess, x, y)
            if i % 100 == 0:
                print("Epoch [%d/%d] | Batch [%d/%d] | Loss:%.2f | LR: %.4f |" % (
                    epoch, args.num_epochs, i, n_batch, loss, lr))
    
        val_loss = []
        for i, (x, y) in enumerate(dl.next_test_batch()):
            loss = model.test_batch(sess, x, y)
            val_loss.append(loss)
        print("<Testing Average Loss>: %.4f" % (sum(val_loss)/len(val_loss)))
    
    n_batch = len(dl.data['submit']['X']) // args.batch_size
    preds = []
    for x in tqdm(dl.next_predict_batch(), total=n_batch, ncols=70):
        preds.append(model.predict_batch(sess, x))
    preds = np.concatenate(preds)

    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    submit = pd.read_csv("../data/sample_submission.csv")
    submit[classes] = preds
    submit.to_csv(model.submit, index=False)
    print(model.submit)


if __name__ == '__main__':
    main()
