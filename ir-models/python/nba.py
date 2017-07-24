import math
import pandas as pd
from scipy.spatial import distance


def closest_player(target_player):
    nba = pd.read_csv('./temp/nba_2013.csv')
    distance_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.',
                        'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

    nba_numeric = nba[distance_columns]
    nba_normalized = (nba_numeric - nba_numeric.mean()) / nba_numeric.std()

    nba_normalized.fillna(0, inplace=True)
    player = nba_normalized[nba["player"] == target_player]

    for dist_fn in [distance.euclidean, distance.cosine]:
        dists = nba_normalized.apply(lambda row: dist_fn(row, player), axis=1)
        dist_frame = pd.DataFrame({'dist': dists})
        dist_frame.sort_values('dist', inplace=True)

        second_smallest = dist_frame.iloc[1].name
        most_similar_to_lebron = nba.loc[second_smallest]['player']
        print("%s - %s   %s" % (target_player, most_similar_to_lebron, str(dist_fn)))
