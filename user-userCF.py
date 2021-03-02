from __future__ import print_function,division
from builtins import range,input

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

#load the data
import os
if not os.path.exists('user2movie.json') or \
    not os.path.exists('movie2user.json') or \
    not os.path.exists('usermovie2rating.json') or \
    not os.path.exists('usermovie2rating_test.json'):
    import preprocess2_dict

with open('user2movie.json','rb') as f:
    user2movie = pickle.load(f)
with open('movie2user.json','rb') as f:
    movie2user = pickle.load(f)
with open('usermovie2rating.json','rb') as f:
    usermovie2rating = pickle.load(f)
with open('usermovie2rating_test.json','rb') as f:
    usermovie2rating_test = pickle.load(f)


N = np.max(list(user2movie.keys())) + 1

m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N: ", N, "M:", M)

if N > 10000:
    print('N = ', N, 'are you sure you want to continue?')
    #exit()


#to find the user similarities, you have to do O(N^2 * M) calculations
#in the "real-world" you'd want to parallelize this
#note: we really only have to do half the calculations, since w_ij is symmetric
K = 25 #number of neighbours
limit = 5 #numero de filmes os usuários precisam ter em comum no mínimo
neighbours = []
averages = []
deviations = []

for i in range(N):
    # ACHAR OS 25 USERS MAIS PROXIMOS DE USER I
    movies_i = user2movie[i]
    movies_i_set = set(movies_i)

    #calculate avg and deviation
    ratings_i = { movie:usermovie2rating[(i,movie)] for movie in movies_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {movie:(rating - avg_i) for movie, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    #save for later use
    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(N):
        #dont include yourself
        if j != i:
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            common_movies = (movies_i_set & movies_j_set) #interseçao

            if len(common_movies) > limit:
                #calcular avg and deviation
                ratings_j = {movie:usermovie2rating[(j,movie)] for movie in movies_j }
                avj_j = np.mean(list(ratings_j.values()))
                dev_j = {movie: (rating - avj_j) for movie, rating in ratings_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
                
                #calcular coeficiente de correlacao

                numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_j)

                #inserir na sorted list e truncar
                #negate weight, because list is sorted ascending
                # maximum value (1) is "closest"
                sl.add((-w_ij,j))
                if len(sl) > K:
                    del sl[-1]
                
    #store the neighbours
    neighbours.append(sl)

    #print out useful things

    if i % 1 == 0:
        print(i)


    #usando neighbours, calculate train and test MSE
def predict(i,m):
    #calculate de weighted sum of deviations
    numerator = 0
    denominator = 0
    #lembrando que o peso esta negativo
    for neg_w, j in neighbours[i]:
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            #o vizinho pode nao ter avaliado o mesmo filme
            #nao quer olhar no dicionario duas vezes
            pass
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    
    prediction = min(5, prediction)
    prediction = max(0.5, prediction) #max rating is 0.5
    return prediction

train_predictions = []
train_targets = []
for (i,m), target in usermovie2rating.items():
    #calculate the predction for this movie
    prediction = predict(i,m)

    #save the prediction and target
    train_predictions.append(prediction)
    train