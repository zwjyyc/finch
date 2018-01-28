import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from data import DataLoader


def main():
    dl = DataLoader()
    model = LogisticRegression()
    
    result = np.zeros((dl.data['submit']['X'].shape[0], dl.data['train']['Y'].shape[1]))
    for i in range(len(dl.params['class_list'])):
        t0 = time.time()
        model.fit(dl.data['train']['X'], dl.data['train']['Y'][:, i])
        print("%.2f secs ==> [%d/6]LogisticRegression().fit()" % (time.time()-t0, i+1))
        result[:, i] = model.predict_proba(dl.data['submit']['X'])[:, 1]

    submit = pd.read_csv("../data/sample_submission.csv")
    submit[dl.params['class_list']] = result
    submit.to_csv('./result.csv', index=False)
    print('End')


if __name__ == '__main__':
    main()
