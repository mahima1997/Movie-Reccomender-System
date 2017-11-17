# Movie-Reccomender-SystemFirst 
I approached the problem with collaborative filtering method which gave Mean Squared Error of 10.14 based on user similarity and MSE of 13.62 based on item similarity.

Limitations of collaborative filtering technique:
Suppose, if I’ve watched ten Romantic movies and you’ve watched ten other ten romantic movies, the raw user action matrix wouldn’t have any overlap. Mathematically, the dot product of our action vectors would be 0. We’d be in entirely separate neighborhoods, even though it seems pretty likely we share at least some underlying preferences.

Using item features (such as genre) could help fix this issue, to a greater extent. Low-Rank Matrix Factorization is that kind of method and SVD i.e Singular Value Decomposition is particularly useful for making recommendations.

So I used SVD to recommend to the specified user top 10 movies of his interest. I reduced my data to k=50 concepts to keep atleast 80% of the energy of my data.
