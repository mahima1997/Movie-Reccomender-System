
"""
Created on Sun Nov 12 00:26:57 2017

@author: mahima gupta
"""
import pandas as pd 
import numpy as np
from scipy.sparse import csc_matrix

links=pd.read_csv("links.csv")

movies=pd.read_csv("movies.csv")

ratings=pd.read_csv("ratings.csv")

tags=pd.read_csv("tags.csv")

train_data=tags.merge(ratings.merge(links.merge(movies,how='outer',on='movieId'),how='outer',on='movieId'),how='outer',on='movieId')
print(ratings.shape)
print(train_data)
data=train_data

#userId_y are the ones giving rating and user_x are the one giving tags
data = data.rename(columns={'userId_x': 'users_giving_tags', 'userId_y': 'users_giving_ratings'})

data_subset=data[['users_giving_ratings','movieId']]
print(data_subset)

table = pd.pivot_table(data,values='rating',index=['users_giving_ratings'],columns=['movieId']).fillna(0.0).as_matrix()
print(table)
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):  #as the first column is user_id
        print(user)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],size=10,replace=False)
        print(test_ratings)
        train[user, test_ratings] = 0
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test
    
def similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        ##the below function helps measure the similarity of every user with every other user 
        squared_sum = ratings.dot(ratings.T) + epsilon     #sum of ratings by combination of users
    elif kind == 'item':
        ##the below function helps measure the similarity of every movie with every other movie
        squared_sum = ratings.T.dot(ratings) + epsilon     #sum of ratings of combination of movies
    sum_root = np.array([np.sqrt(np.diagonal(squared_sum))])  
    print("sum_root",sum_root)
    print(sum_root.shape)
    ##the below line actually calculates the similarity lying between 0 to 1
    return (squared_sum / sum_root / sum_root.T)

 
def predict(ratings, similarity, kind='user'):
    if kind == 'user':
        #this function predicts what rating a user would give to a movie depending on user similarity matrix.
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        #this function also predicts what rating a user would give to a movie depending on item similarity matrix.
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    #############genres not used to improve predictions##############

train, test = train_test_split(table)
user_similarity = similarity(train, kind='user')
item_similarity = similarity(train, kind='item')    
    
from sklearn.metrics import mean_squared_error

#to calculate the mean squared error
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)  

item_prediction = predict(train, item_similarity, kind='item')  #this will give ratings a user would give a movie
                                                                #depending on what movies of that kind he himself has
                                                                #seen
user_prediction = predict(train, user_similarity, kind='user')  #this will give ratings a user would give a movie
                                                                #depending on what movies users of his kind see

print ('User-based CF MSE: ' + str(get_mse(user_prediction, test)))  #Collaborative Filtering Mean Squared Error
print ('Item-based CF MSE: ' + str(get_mse(item_prediction, test)))


