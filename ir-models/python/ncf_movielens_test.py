import pandas as pd
from ncf import recommend


def favourite(active_user, top_n, rating_info, movie_info):
    df = rating_info[rating_info.userid == active_user]
    top_ratings = df.sort_values(['rating'], ascending=[0])[:top_n]
    top_movies = movie_info.loc[movie_info.itemid.isin(top_ratings.itemid)]
    return list(top_movies.title)


def main():
    rating_info = pd.read_csv('./temp/ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2],
                              names=['userid', 'itemid', 'rating'])

    rating_matrix = pd.pivot_table(rating_info, values='rating', index=['userid'], columns=['itemid'])

    movie_info = pd.read_csv('./temp/ml-100k/u.item', sep='|', header=None, index_col=False,
                             names=['itemid', 'title'], usecols=[0,1], encoding='latin')

    print(favourite(5, 10, rating_info, movie_info), end='\n\n')
    
    print(recommend(5, 10, rating_matrix, movie_info))


if __name__ == '__main__':
    main()
