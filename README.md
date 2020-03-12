# MovieRecommendationSystem
This recommendation system uses a variety of recommendation algorithms implemented from scratch

Training Data
The training data: a set of movie ratings by 200 users (userid: 1-200) on 1000 movies (movieid: 1-1000). 
The data is stored in a 200 row x 1000 column table. Each row represents one user. 
Each column represents one movie. A rating is a value in the range of 1 to 5, where 1 is "least favored" and 
5 is "most favored". Value 0 zero means user has not rated the movie

Test Data
 A pool of movie ratings by 100 users (userid: 201-300). Each user has already rated 5 movies. 
 The format of the data is as follows: the file contains 100 blocks of lines. 
 Each block contains several triples : (U, M, R), which means that user U gives R points to movie M. 
 Please note that in the test file, if R=0, then you are expected to predict the best possible rating 
 which user U will give movie M. The following is a block for user 276. (line 6545-6555 of test5.txt)
    276 42 4 // user 276 gives movie 42 4 points. 
    276 85 2 // user 276 gives movie 85 2 points.

Algorithms implemented
1. User based collaborative filtering
Cosine similarity, Pearson Correlation
Extensions: Inverse User Frequency, Case Modification
2. Item based collaborative filtering
3. Ensamble

Evaluation:
RSME

