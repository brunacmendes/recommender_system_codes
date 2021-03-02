import pandas as pd 

df = pd.read_csv('dataset/ml-25m/ratings.csv') 

#note:
#user ids are ordered sequentially from 1 with no missing numbers
#movie ids are integers
#not all movie ids appear

#fazer os users ids irem de 0...N-1
df.userId = df.userId - 1

#criar um mapa para movie id
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
	movie2idx[movie_id] = count
	count +=1


#add them to the dataframe
df["movie_idx"] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('dataset/ml-25m/edited_ratings.csv')