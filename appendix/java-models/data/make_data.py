import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


X, y = make_classification()
data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
pd.DataFrame(
    data = data,
    columns = ['feature'+str(i) for i in range(20)] + ['target']
).to_csv('./data.csv', index=False)
