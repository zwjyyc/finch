import pandas as pd
import numpy as np
import pprint

from scipy.special import logit, expit


submissions = [
    'model_gru/result.csv',
    'model_lstm/result.csv',
    'model_tfidf_lr/result.csv',
]

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def main():
    submit = pd.read_csv("./data/sample_submission.csv")
    shape = submit[labels].values.shape
    result = np.zeros(shape)

    for submission in submissions:
        result += pd.read_csv(submission)[labels].values
    result /= len(submissions)

    """
    the post-processing issue, please check here:
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48364
    """
    #result = result ** 1.4
    result = expit(logit(result)-0.5)
    
    submit[labels] = result
    submit.to_csv('./final.csv', index=False)
    print("Ensemble Finished")
    pprint.PrettyPrinter().pprint(submissions)


if __name__ == '__main__':
    main()