from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as func
from pyspark.mllib.recommendation import ALS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#set spark context
conf = SparkConf().setAppName("NMFRecommendation")
sc = SparkContext(conf=conf)
sc.setLogLevel("Error")
sqlContext = SQLContext(sc)


#artistData = sc.textFile('F:\courses\Spring_2017\COSC_526_Big_Data_Mining\Assignment_1\Artist_data')
#artistIds = artistData.map(lambda line:line.strip().split(' ')[0])


def createArtistMap(artist_data):    
    artist_data_list = []
    for line in artist_data.collect():
        key = line[0]
        if key.isdigit():
            artist = ' '.join(line[1:])        
            artist_data_list.append((float(key), artist))            

    return sc.parallelize(artist_data_list)

print "reading artist_data:"
artist_data = sc.textFile('file:///F:/courses/Spring_2017/COSC_526_Big_Data_Mining/Assignment_1/_artist_data')

print "this is what artist data looks like:"
print artist_data


artist_data = artist_data.map(lambda line:line.encode("ascii", "ignore").strip().split()).filter(lambda line: len(line) > 1)

print "calling artist data:"
artist_data = createArtistMap(artist_data)
artist_dataDF = artist_data.toDF(['artist_id', 'artist_name'])

user_artist_dataRDD = sc.textFile('F:\courses\Spring_2017\COSC_526_Big_Data_Mining\Assignment_1\User_artist_data')
user_artist_dataRDD = user_artist_dataRDD.map(lambda line:line.encode("ascii", "ignore").strip().split()).map(lambda line: [float(x) for x in line])
artist_aliasRDD = sc.textFile('F:\courses\Spring_2017\COSC_526_Big_Data_Mining\Assignment_1\Artist_alias')
artist_aliasRDD = artist_aliasRDD.map(lambda line:line.encode("ascii", "ignore").strip().split()).map(lambda line: [float(x) for x in line])

#Some rows in artist alias have just 1 element. Need to remove those else the join later fails
user_artist_dataRDD = user_artist_dataRDD.filter(lambda line: len(line) > 2 and len(line) < 4)
artist_aliasRDD = artist_aliasRDD.filter(lambda line: len(line) > 1 and len(line) < 3)


#Figure out how to map these two arrays

print "paralellizing:"
user_artist_dataList = []
for x in  user_artist_dataRDD.take(500):
    user_artist_dataList.append(x)
user_artist_dataRDD = sc.parallelize(user_artist_dataList)
print "done paralellizing:"


user_artist_dataDF = user_artist_dataRDD.toDF(['user', 'artist', 'count'])
artist_aliasDF = artist_aliasRDD.toDF(['artist1', 'alias'])


print "Joining data"
finalData = user_artist_dataDF.join(artist_aliasDF, user_artist_dataDF['artist'] == artist_aliasDF['artist1'])

for line in finalData.rdd.collect():
    print line

finalData = finalData.drop('artist1')

finalData = finalData.groupBy("user","alias").agg(func.sum("count").alias("count"))

print "After grouping:"

for line in finalData.rdd.collect():
    print line

ratings = finalData.rdd
print "working on initial model:"
model = ALS.train(ratings , 10, 5, 0.01, 1)
predictRatings = ratings.map(lambda line: (line[0], line[1]))
predictions = model.predictAll(predictRatings).map(lambda x: (x[0], x[1], x[2]))
predictionsDF = predictions.toDF(['user', 'predicted_artist_id', 'count'])

finalPredictedUserArtistName = predictionsDF.join(artist_dataDF, predictionsDF['predicted_artist_id'] == artist_dataDF['artist_id'])

print finalPredictedUserArtistName
print "printing final predictions:"
for line in finalPredictedUserArtistName.collect():
    print line

print "done train test split:"

####90-10 train test data split
finalData = finalData.randomSplit([0.9, 0.1])
trainData = finalData[0].rdd
testData = finalData[1].rdd

#Training the model on train data set
model = ALS.train(trainData, 10, 5, 0.01, 1)

#Cross validation to study recommeder system
print("done training model, now testing:")
testDataUserArtist = testData.map(lambda x: (x[0], x[1]))
predictions = model.predictAll(testDataUserArtist).map(lambda r: ((r[0], r[1]), r[2]))
testDataPredictions = testData.map(lambda r:((r[0], r[1]), r[2])).join(predictions)
MSE = testDataPredictions.map(lambda x:(x[1][0] - x[1][1])**2).mean()

print("Mean squared error is:" + str(MSE))

