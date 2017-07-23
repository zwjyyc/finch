import pandas as pd
from ncf import recommender


def favourite(active_user, top_n, df, movie_info):
    top_movies = pd.DataFrame.sort_values(df[df.userid==active_user], ['rating'], ascending=[0])[:top_n]
    top_movies = movie_info.loc[movie_info.itemid.isin(top_movies.itemid)]
    return list(top_movies.title)


def main():
    rating_info = pd.read_csv('./temp/ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2],
                    names=['userid', 'itemid', 'rating'])

    rating_matrix = pd.pivot_table(rating_info, values='rating', index=['userid'], columns=['itemid'])

    movie_info = pd.read_csv('./temp/ml-100k/u.item', sep='|', header=None, index_col=False,
                             names=['itemid', 'title'], usecols=[0,1], encoding='latin')

    print(favourite(5, 10, rating_info, movie_info))
    print()
    print(recommender(5, 10, rating_matrix, movie_info))


if __name__ == '__main__':
    main()
