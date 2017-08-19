from apriori import Apriori
import pandas as pd


def main():
    df = pd.read_csv('./temp/ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2], names=['userid', 'itemid', 'rating'])
    table = pd.pivot_table(df, values='rating', index=['userid'], columns=['itemid'])
    model = Apriori()
    model.fit(table)
    model.predict()


if __name__ == '__main__':
    main()
