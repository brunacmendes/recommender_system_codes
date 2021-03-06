
#spark-submit --master spark://localhost:7077 spark2.py


#SETUP DO CLUSTER AULA 72 (não fiz)

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext

# increase memory
# SparkContext.setSystemProperty('spark.driver.memory', '10g')
# SparkContext.setSystemProperty('spark.executor.memory', '10g')

sc = SparkContext("local", "Your App Name Here")


# load in the data
# data = sc.textFile("../large_files/movielens-20m-dataset/small_ratings.csv")
data = sc.textFile("dataset/ml-25m/ratings.tar.gz")

# filter out header
header = data.first() #extract header
data = data.filter(lambda row: row != header)

# convert into a sequence of Rating objects
ratings = data.map(
  lambda l: l.split(',')
).map(
  lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))
)

# split into train and test
train, test = ratings.randomSplit([0.8, 0.2])

# train the model
K = 10
epochs = 10
model = ALS.train(train, K, epochs)

# evaluate the model

# train
x = train.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(p)
# joins on first item: (user_id, movie_id)
# each row of result is: ((user_id, movie_id), (rating, prediction))
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("***** train mse: %s *****" % mse)


# test
x = test.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(p)
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("***** test mse: %s *****" % mse)

#para pegar o resultado das predicoes

#p = predictions

#p.saveAsTextFile(path_to_output)

#COMO USAR AS PREDIÇÕES NA MINHA APLICAÇÃO: aula 73