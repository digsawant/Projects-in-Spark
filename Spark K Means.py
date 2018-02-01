from pyspark import SparkConf, SparkContext
from collections import OrderedDict
from pyspark.mllib.clustering import KMeans
from math import sqrt


#set spark context
conf = SparkConf().setAppName("KDData")
sc = SparkContext(conf=conf)
sc.setLogLevel("Error")

#load data
rawData = sc.textFile('F:\courses\Spring_2017\COSC_526_Big_Data_Mining\Assignment_1\kddcup.data_10_percent_corrected')
rawData = rawData.map(lambda line: line[:-1])
labels = rawData.map(lambda line: line.strip().split(',')[-1])

labels_list = []
for x in labels.collect():
    labels_list.append(x)


label_counts = labels.countByValue()

sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t:t[1], reverse=True))


def searchKeyValuePairs(elem):
    return keyValue[elem]

keyValue = {}

#Print label counts
for label, count in sorted_labels.items():
    keyValue[label.encode("ascii", "ignore")] = count
    print label, count


#Cleaning data

#Stripping middle 3 categorical elements
cleanedData = rawData.map(lambda line: line.strip().split(',')[0] + ',' + ','.join(line.strip().split(',')[4:]))

 
finalData = []
finalDataWithIdentifier = []

#Really bad code, needs work but does the job
for elem in cleanedData.collect():
    midElem = elem.encode("ascii", "ignore").split(',')
    midElem[-1] = searchKeyValuePairs(midElem[-1])
    finalList = []
    for f in midElem:
        finalList.append(float(f)) 
    finalDataWithIdentifier.append(finalList)
    finalList = finalList[0:37]
    finalData.append(finalList)
   
#End of cleaning data
        
finalData = sc.parallelize(finalData)

#### Choosing K, calculating clustering score
def calc_dist(point, clusters):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))



#Calculate K-means, indvidual distances of points, print points assigned to each cluster
def calculateKMeans(i, labels_list):
    clusters = KMeans.train(finalData, i, maxIterations=10, initializationMode="random")
    Euc_distance = finalData.map(lambda point: calc_dist(point, clusters)).reduce(lambda x, y: x + y)/i  
    print("for " + str(i) + " clusters the average Euc_distance is:" + str(Euc_distance))
    if(i == 175):
        Indiv_Euc_distance = finalData.map(lambda point: calc_dist(point, clusters)) 
        clusterAssignments = finalData.map(lambda point:clusters.predict(point)).cache()
        assignmentList = []
        for x in  clusterAssignments.collect():
            assignmentList.append(x)
        labels_listRDD = sc.parallelize(labels_list)
        assignmentList = sc.parallelize(assignmentList)
        rdd = labels_listRDD.zip(assignmentList).map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).map(lambda ((x1,x2),y): (x1,x2,y))
        print "printing assignments:"
        for x,y,z in rdd.collect():
            print "" + str(y) + " " + str(x) + " " + str(z)
        Indiv_Euc_distance_list = []
        for item in Indiv_Euc_distance.collect():
            Indiv_Euc_distance_list.append(item)        
        return sc.parallelize(Indiv_Euc_distance_list)
    return 0
 


#Calculate the Euc_distance from 5 to 180
for i in range(5, 180, 10):
    if(i == 175):
        Indiv_Euc_distanceRDD = calculateKMeans(i, labels_list)
    else:
        calculateKMeans(i, labels_list)

#Zip cleanedData with their distances
cleanedDataRDD = []
for x in  cleanedData.collect():
    cleanedDataRDD.append(x)
cleanedDataRDD = sc.parallelize(cleanedDataRDD)

#Getting top 10 farthest points and interpreting them
distanceList = Indiv_Euc_distanceRDD.zip(cleanedDataRDD)
print(distanceList.takeOrdered(10, key = lambda x: -x[0]))