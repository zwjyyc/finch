import pandas as pd
from ncf import recommender


def main():
    df = pd.read_csv('./temp/ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2],
                    names=['userid', 'itemid', 'rating'])

    rating_matrix = pd.pivot_table(df, values='rating', index=['userid'], columns=['itemid'])

    movie_info=pd.read_csv('./temp/ml-100k/u.item', sep='|', header=None, index_col=False,
                        names=['itemid', 'title'], usecols=[0,1], encoding='latin')

    print(recommender(5, 3, rating_matrix, movie_info))


if __name__ == '__main__':
    main()
