
"""
Created on Sun Nov 15 00:26:57 2017

@author: mahima gupta
"""

#There are many different ways to factor matrices, but singular value decomposition is particularly useful for making recommendations.

import pandas as pd
#used for matrix factorization
from scipy.sparse.linalg import svds  #used for matrix factorization

movies=pd.read_csv("movies.csv")
print(movies.genres.unique())

import pandas as pd
import numpy as np

ratings_list = pd.read_csv("ratings.csv")
#users_list = [i.strip().split("::") for i in open('users.dat', 'r').readlines()]
movies_list = pd.read_csv("movies.csv")

ratings_df = pd.DataFrame(ratings_list)#, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies_df = pd.DataFrame(movies_list)#, columns = ['MovieID', 'Title', 'Genres'])

print(movies_df.head())
print(ratings_df)

R_df = pd.pivot_table(ratings_df, index = ['userId'], columns =['movieId'], values ='rating').fillna(0.0)
print(R_df.head())
R = R_df.as_matrix()

#calculating the mean rating of users
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R-user_ratings_mean.reshape(-1, 1)

#Compute SVD of the input user ratings matrix
U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

def recommend_movies(predictions_df, userId, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userId - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userId)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False))

    print 'User {0} has already rated {1} movies.'.format(userId, user_full.shape[0])
    print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',left_on = 'movieId',
               right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values
                       ('Predictions', ascending = False).iloc[:num_recommendations, :-1])

    return user_full, recommendations

already_rated, predictions = recommend_movies(preds_df, 437, movies_df, ratings_df, 10)
print(predictions)



