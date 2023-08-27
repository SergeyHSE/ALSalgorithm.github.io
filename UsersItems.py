# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:53:38 2023

@author: SergeyHSE
"""
#############################################################################################
#    In this work, we will find similar movies and users using the ALS algorithm,
#    implement the calculation of the NDCG metric, and investigate
#    the effect of the dimensionality of hidden representations on the performance of the algorithm.
#    Dataset = MovieLens
#############################################################################################


import zipfile
from collections import defaultdict, Counter
import datetime
from scipy import linalg
import numpy as np
import pandas as pd

#Let's unpack the data and see how it's organized.


path = r"Your path.zip"
#If you use Linux we shouldn't apply 'replace'
path = path.replace('\\', '/')

with zipfile.ZipFile(path, "r") as z:
    print("files in archive")
    print(z.namelist())
    print("movies")
    with z.open("ml-1m/movies.dat") as m:
        print(str(m.readline()).split("::"))
    print("users")
    with z.open("ml-1m/users.dat") as m:
        print(str(m.readline()).split("::"))
    print("ratings")
    with z.open("ml-1m/ratings.dat") as m:
        print(str(m.readline()).split("::"))

zip = zipfile.ZipFile(path)
file = zip.read('ml-1m/README')
print(file)

# We can see that the archive contains information about movies.
# This is the movieId of the movie, title and genre.
# About users we know userId, gender (F, M), age, coded employment information and zip-code.
# And the rating information:
#   userId, movieId, rating and the moment in time when the rating was made.
# Let's read the data.

movies = {} # id
users = {} # id
ratings = defaultdict(list) # user-id

with zipfile.ZipFile(path, "r") as z:
    # parse movies
    with z.open("ml-1m/movies.dat") as m:
        for line in m:
            MovieID, Title, Genres = line.decode('iso-8859-1').strip().split("::")
            MovieID = int(MovieID)
            Genres = Genres.split("|")
            movies[MovieID] = {"Title": Title, "Genres": Genres}
            
    
    # parse users
    with z.open("ml-1m/users.dat") as m:
        fields = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
        for line in m:
            row = list(zip(fields, line.decode('iso-8859-1').strip().split("::")))
            data = dict(row[1:])
            data["Occupation"] = int(data["Occupation"])
            users[int(row[0][1])] = data
    
    # parse ratings
    with z.open("ml-1m/ratings.dat") as m:
        for line in m:
            UserID, MovieID, Rating, Timestamp = line.decode('iso-8859-1').strip().split("::")
            UserID = int(UserID)
            MovieID = int(MovieID)
            Rating = int(Rating)
            Timestamp = int(Timestamp)
            ratings[UserID].append((MovieID, Rating, datetime.datetime.fromtimestamp(Timestamp)))
    
len(users)
len(ratings)
len(movies)

#Now let's split the data into training and test samples
#We will split the ratings by time.
#First, let's find the date on which 80% of the ratings were set in the dataset.
#And all the ratings that were before will go to train, and the rest to test.

times = []
for user_ratings in ratings.values():
    times.extend([x[2] for x in user_ratings])
times = sorted(times)

threshold_time = times[int(0.8 * len(times))]
threshold_time

train = []
test = []

for user_id, user_ratings in ratings.items():
    train.extend((user_id, rating[0], rating[1] / 5.0) for rating in user_ratings if rating[2] <= threshold_time)
    test.extend((user_id, rating[0], rating[1] / 5.0) for rating in user_ratings if rating[2] > threshold_time)
print("ratings in train:", len(train))
print("ratings in test:", len(test))


train_by_user = defaultdict(list)
test_by_user = defaultdict(list)
for u, i, r in train:
    train_by_user[u].append((i, r))
for u, i, r in test:
    test_by_user[u].append((i, r))

train_by_item = defaultdict(list)
for u, i, r in train:
    train_by_item[i].append((u, r))

#Let's count the number of users and movies

n_users = max([e[0] for e in train]) + 1
n_items = max([e[1] for e in train]) + 1
n_users
n_items

################################################
#                     ALS
################################################
np.random.seed(0)
LATENT_SIZE = 10
N_ITER = 20

# regularizers
lambda_p = 0.2
lambda_q = 0.001

# latent representations
p = 0.1 * np.random.random((n_users, LATENT_SIZE))
q = 0.1 * np.random.random((n_items, LATENT_SIZE))

#Now let us compose a matrix P from vectors p_u and a matrix Q from vectors q_i .
#By matrix Q(see 'FormulaQ.jpg' in files of this repository)  we denote the submatrix of matrix Q only for goods evaluated by user u ,
#where n_u is the number of evaluations of user u .
#The reconfiguration step pu for a fixed matrix Q reduces to ridge regression tuning and looks like this:
#See 'FormulaRidge.jpg

def compute_p(p, q, train_by_user):
    for u, rated in train_by_user.items():
        rated_items = [i for i, _ in rated]
        rated_scores = np.array([r for _, r in rated])
        Q = q[rated_items, :]
        A = (Q.T).dot(Q)
        d = (Q.T).dot(rated_scores)
        p[u, :] = np.linalg.solve(lambda_p * len(rated_items) * np.eye(LATENT_SIZE) + A, d)
    return p

def compute_q(p, q, train_by_item):
    for i, rated in train_by_item.items():
        rated_users = [j for j, _ in rated]
        rated_scores = np.array([s for _, s in rated])
        P = p[rated_users, :]
        A = (P.T).dot(P)
        d = (P.T).dot(rated_scores)
        q[i, :] = np.linalg.solve(lambda_q * len(rated_users) * np.eye(LATENT_SIZE) + A, d)
    return q

def train_error_mse(predictions):
    return np.mean([(predictions[u, i] - r) ** 2 for u, i, r in train])

def test_error_mse(predictions):
    return np.mean([(predictions[u, i] - r) ** 2 for u, i, r in test])


for iter in range(N_ITER):
    p = compute_p(p, q, train_by_user)
    q = compute_q(p, q, train_by_item)

    predictions = p.dot(q.T)
    
    print(iter, train_error_mse(predictions), test_error_mse(predictions))

#############################################################
#     Calculate third movies similared 'Star WAr (1980)
#############################################################

# Search for the movie title in the dictionary
search_title = 'Star Wars: Episode V - The Empire Strikes Back (1980)'

found_movie = None
for movie_id, movie_info in movies.items():
    if movie_info['Title'] == search_title:
        found_movie = {movie_id: movie_info}

print(found_movie)

p.shape
q.shape

#Let's calculate the scalar product of its embedding with the rest of the movie

from sklearn.metrics.pairwise import cosine_similarity

movie_cos_simiraity = cosine_similarity(q)
movie_cos_simiraity.shape

movie_star_war = movie_cos_simiraity[1196]
movie_star_sorted = np.argsort(movie_star_war)[::-1]

first_movies = 10
print(f'Previous {first_movies} movies least similar to Movie {movie_star_war}:')
cosine = []
for i in range(1, first_movies + 1):
    similar_movie_index = movie_star_sorted[i]
    similarity = movie_star_war[similar_movie_index]
    print(f'Movie {similar_movie_index}: Similarity = {similarity:.4f}')
    cosine.append((similarity))

answer1 = sum(cosine[:3])
print(answer1)

#found names all similar movies

found_mov = None
for movie_id, movie_info in movies.items():
    if movie_id == 260:
        found_mov = {movie_id: movie_info}
        print(found_mov)
    elif movie_id == 1210:
        found_mov = {movie_id: movie_info}
        print(found_mov)
    elif movie_id == 1198:
        found_mov = {movie_id: movie_info}
        print(found_mov)
    elif movie_id == 1196:
        found_mov = {movie_id: movie_info}
        print(found_mov)
    
#Let's build correlation matrix

import matplotlib.pyplot as plt

np_items = np.array(movie_cos_simiraity)
np_items = np.corrcoef(np_items)

fig, ax = plt.subplots(figsize=(18,18))
heatmap = ax.imshow(np_items, cmap='Paired')
for i in range(np_items.shape[0]):
    ax.text(i, i, "1.0", ha="center", va="center", color="white",
            fontsize=2, fontweight='light')
cbar = ax.figure.colorbar(heatmap, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom", fontsize=18)
plt.title('Correlation between ITEM and ITEM', fontsize=18)
plt.show()

#See picture 'correlation.png'

#Find film with minimum correlation movies

np_items = pd.DataFrame(np_items)
min_corr = np_items.min().min()

min_corr_row = np_items[np_items == min_corr].stack().index[0][0]
min_corr_col = np_items[np_items == min_corr].stack().index[0][1]

print(f"Minimum Correlation Coefficient: {min_corr}")
print(f"Correlation between {min_corr_row} and {min_corr_col}")

min_cor_two = None
for movie_id, movie_info in movies.items():
    if movie_id == 823:
        min_cor_two = {movie_id: movie_info}
        print(min_cor_two)
    elif movie_id == 2955:
        min_cor_two = {movie_id: movie_info}
        print(min_cor_two)
        

correlation_totals = np_items.abs().sum(axis=1)

min_corr_row = correlation_totals.idxmin()

print(f"The row with the minimum total correlation is '{min_corr_row}' with a total correlation value of {correlation_totals[min_corr_row]}")

min_cor_movie = None
for movie_id, movie_info in movies.items():
    if movie_id == 1796:
        min_cor_movie = {movie_id: movie_info}
        print(min_cor_movie)


#figure out what id missing in 'Movies' and if whether it have influence on the results

with zipfile.ZipFile(path, "r") as zi:
    with zi.open("ml-1m/movies.dat") as h:
        movies_id = []
        for line in h:
            ID, T, G = line.decode('iso-8859-1').strip().split("::")
            movies_id.append(ID)
            
len(movies_id) 
dir(movies_id)
movies_ids = []       
for i in movies_id:
    i = int(i)
    movies_ids.append(i)

#Find wich variables is missing of
total_id = 3952
existing_id = set(movies_ids)
missing_id = [id for id in range(1, total_id + 1) if id not in existing_id]
print("Missing IDs:", missing_id)
type(missing_id)

type(q)

mask = np.zeros(q.shape[0], dtype=bool)
mask[missing_id] = True
rows_to_keep = ~mask             
q_filtered = q[rows_to_keep]
q_filtered.shape

# CAlculate how many ids before Stars War was deleted
count_items = 0
for item in missing_id:
    
    if item < 1196:
        count_items += 1
        continue
    elif item >= 1196:
        break
print(count_items)


movie_cos_sim_filtered = cosine_similarity(q_filtered)
movie_cos_sim_filtered.shape

movie_star_war_filter = movie_cos_sim_filtered[(1196-count_items)]
movie_star_sorted_filter = np.argsort(movie_star_war_filter)[::-1]

first_movies_filter = 10
print(f'Previous {first_movies_filter} movies least similar to Movie {movie_star_war_filter}:')
cosine_filter = []
for i in range(1, first_movies_filter + 1):
    similar_movie_index_filter = movie_star_sorted_filter[i]
    similarity_filter = movie_star_war_filter[similar_movie_index_filter]
    print(f'Movie {similar_movie_index_filter}: Similarity = {similarity_filter:.4f}')
    cosine_filter.append((similarity_filter))

answer1_filter = sum(cosine_filter[:3])
print(answer1_filter)
print(answer1) #Answers match

#Check all films with strating 'Star Wars...'

search_prefix = 'Star Wars'

found_movi = {}
for movie_id, movie_info in movies.items():
    if movie_info['Title'].startswith(search_prefix):
        found_movi[movie_id] = movie_info

print(found_movi)

# We end up with several similar movies, one of which is not similar in title to "Star Wars...."
# but has more in common than a movie that came out a few decades later

######################################################################
#  Calculate number of estemated movies for two similar users,
#           where one of them have the id=5472 
######################################################################

users[5472]
rate = pd.DataFrame(ratings[5472])
rate
rate2 = pd.DataFrame(ratings[5009])
rate2

p.shape
user_cosine_sim = cosine_similarity(p)
user_cosine_sim.shape

user_cos_id = user_cosine_sim[5472]
user_cos_id
user_sorted = np.argsort(user_cos_id)[::-1]

first_users = 10
print(f'Previous {first_users} users least similar to user {user_cos_id}:')

for i in range(0, first_users + 1):
    similar_user_idx = user_sorted[i]
    similarity_user = user_cos_id[similar_user_idx]
    print(f'User {similar_user_idx}: Similarity = {similarity_user:.5f}')

ratings[5472]    
ratings[5009]

count_oneuser = 0
count_twouser = 0
for id_one, number_of_rate in ratings.items():
    if id_one == 5472:
        for k in number_of_rate:
            count_oneuser += 1
        print('Count of one:', count_oneuser)
    elif id_one == 5009:
        for k in number_of_rate:
            count_twouser += 1
        print('Count of two:', count_twouser)
print('Total number of watched movies:', count_oneuser+count_twouser)

#Build correlation matrix

np_user = np.corrcoef(np.array(user_cosine_sim))
np_user = pd.DataFrame(np_user)

fig, ax = plt.subplots(figsize=(18,18))
heatmap = ax.imshow(np_user, cmap='Paired')
for i in range(np_user.shape[0]):
    ax.text(i, i, "1.0", ha="center", va="center", color="black",
            fontsize=2, fontweight='light')
cbar = ax.figure.colorbar(heatmap, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom", fontsize=18)
plt.title('Correlation between USER and USER', fontsize=18)
plt.show()
# See 'correlation_users.png'

# If we compare both pictures (correlation matrices) with each other,
#we can see that the users' matrix is smoother. It can be assumed that this is due to the big difference between
#the movies, but it is not obvious. Perhaps it is due to the weak dispersion of the scores.
#The point is that users are initially represented with similar characteristics,
#which gives a weak variance in the correlation matrix, while movies already have unique features.

################################################################
#               relizing DCG and NDSG metrics
################################################################

predictions = p.dot(q.T)
predictions.shape
prep = predictions[1]
prep.shape

#Firstly let's calculate DCG for a litle random list of numbers and see how it works

scores = [5, 5, 4, 5, 2, 4, 5, 3, 5, 5, 2, 3, 0, 0, 1, 2, 2, 3, 0]
scores = np.array(scores)

def DCG_k(scores, k):

    if k > len(scores):
        k = len(scores)
    discounts = np.log2(np.arange(k) + 2)
    return np.sum((2**scores[:k] - 1) / discounts)

def NDCG_k(scores, k, ideal_scores=None):
    
    if ideal_scores is None:
        ideal_scores = np.sort(scores)[::-1]
        
    dcg_at_k = DCG_k(scores, k)
    ideal_dcg_at_k = DCG_k(ideal_scores, k)
    
    if ideal_dcg_at_k == 0:
        return 0.0
    
    return dcg_at_k / ideal_dcg_at_k

print('Answer for third qustion:', NDCG_k(scores, k=5))

# Calculate DCG and NDCG on all our train and test data

predictions.shape
predictions = predictions.T
predictions.shape[1]


def DCG(scores):
    k = len(scores)
    discounts = np.log2(np.arange(k) + 2)
    return np.sum((2**scores[:k] - 1) / discounts)

def NDCG(scores, ideal_scores=None):
    
    if ideal_scores is None:
        ideal_scores = np.sort(scores)[::-1]
        
    dcg_at_k = DCG(scores)
    ideal_dcg_at_k = DCG(ideal_scores)

    if ideal_dcg_at_k == 0:
        return 0.0
    
    return dcg_at_k / ideal_dcg_at_k

ndcg_scores = []

for query_scores in predictions:
    ndcg = NDCG(query_scores)
    ndcg_scores.append(ndcg)
    
average_ndcg = np.mean(ndcg_scores)
print("Average NDCG:", average_ndcg)

#Now let's build predictions figure for fun

import matplotlib.colors as mcolors

norm = mcolors.Normalize(vmin=0, vmax=predictions.max())

fig, ax = plt.subplots(figsize=(18,18))
heatmap = ax.imshow(predictions, cmap='tab20c', norm=norm)
cbar = ax.figure.colorbar(heatmap, ax=ax, ticks=np.linspace(0, predictions.max()), format='%.2f')
cbar.ax.set_ylabel('Astimates', rotation=-90, va="bottom", fontsize=18)
plt.title('Prediction astimates', fontsize=18)
plt.xlabel('Users', fontsize=18)
plt.ylabel('Movies', fontsize=18)
plt.show()

# We see seek correlation between first 600 users and movies, but then this picture change and it very intresting.
