from __future__ import print_function
from datetime import datetime
from dis import disco
import os
import numpy as np
import sys
import re
from turtle import distance
import requests
from operator import add
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import math
from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC,LinearSVCModel,DecisionTreeClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT,DenseVector
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import *
from datetime import *
from sklearn.tree import DecisionTreeRegressor
import functools

###########################################
#------------Please find this note from my data source below------
#
#Recipients of Retrosheet data are free to make any desired use of
#the information, including (but not limited to) selling it,
#giving it away, or producing a commercial product based upon the
#data.  Retrosheet has one requirement for any such transfer of
#data or product development, which is that the following
#statement must appear prominently:
#
#     The information used here was obtained free of
#     charge from and is copyrighted by Retrosheet.  Interested
#     parties may contact Retrosheet at "www.retrosheet.org".
#
#Retrosheet makes no guarantees of accuracy for the information 
#that is supplied. Much effort is expended to make our website 
#as correct as possible, but Retrosheet shall not be held 
#responsible for any consequences arising from the use the 
#material presented here. All information is subject to corrections 
#as additional data are received. We are grateful to anyone who
#discovers discrepancies and we appreciate learning of the details. 
##########################################



spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
#sc = SparkContext(appName="Hw3", conf=SparkConf().set('spark.driver.memory', '24g').set('spark.executor.memory', '12g'))
sqlContext = SQLContext(sc)

#----define a cleaner for the data----
#mkaes sure that every value is filled and we have complete data for a game that is wanetd
def correctRows(p):
    if(len(p)==35):
        return p
#----joins two dataframes together by a union----
#https://www.geeksforgeeks.org/merge-two-dataframes-in-pyspark/
def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)

#----read in data----
#local
gamedata2021 = r"C:\Users\jamesbloor2\Desktop\BU\CS777\2021_gamelog_data.csv"
gamedata2021 = sc.textFile(gamedata2021)
#online
#corpus = sc.textFile(sys.argv[1])
gamedata2021 = gamedata2021.map(lambda x: x.split(','))
#count = gamedata.count()
#print(count)
#print(gamedata.take(2))

#----Clean data up----
#3,4,5,7,8,13,17,18,78,80,82,84,90,92,102,104,106,109,112,115,118,121,124,127,130,133,136,139,142,145,148,151,154,157,10,11
#select columns wanted
mlbgamedata = gamedata2021.map(lambda x: (x[3-1], x[4-1]+"V",x[5-1]+"V",x[7-1]+"H",x[8-1]+"H",x[13-1],x[17-1],x[78-1],x[80-1],x[82-1],x[84-1]\
    ,x[90-1]+"V",x[92-1]+"H",x[102-1]+"V",x[104-1]+"H",x[106-1]+"V",x[109-1]+"V",x[112-1]+"V",x[115-1]+"V",x[118-1]+"V",x[121-1]+"V",x[124-1]+"V"\
        ,x[127-1]+"V",x[130-1]+"V",x[133-1]+"H",x[136-1]+"H",x[139-1]+"H",x[142-1]+"H",x[145-1]+"H",x[148-1]+"H",x[151-1]+"H",x[154-1]+"H",x[157-1]+"H",x[10-1],x[11-1]))
#print(mlbgamedata.take(3))

#make sure data is fully filled
mlbgamedataCleaned = mlbgamedata.filter(correctRows)

#how much data was filtered out?
countBeforeClean = mlbgamedata.count()
countAfterClean = mlbgamedataCleaned.count()
print("2021 Count before clean:", countBeforeClean,"\n", "2021 Count after clean:",countAfterClean)

#----make my own columns that I will need----
#1 will be a home win
#0 will be a visitor win 
mlbGameWL = mlbgamedataCleaned.map(lambda x: (1 if (x[34]>x[33]) else 0, (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],\
    x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],x[23],x[24],x[25],x[26],x[27],x[28],x[29],x[30],x[31],\
        x[32])))

#print(mlbGameWL.take(3))

schema = StructType([ 
    StructField("label",IntegerType(),True), 
    StructField("features",ArrayType(StringType(),True))])
DFmlbGameWL2021 = spark.createDataFrame(data=mlbGameWL,schema=schema).cache()

cv = CountVectorizer(inputCol="features", outputCol="featuresVector")
cvmodel = cv.fit(DFmlbGameWL2021)
DFmlbGameWL = cvmodel.transform(DFmlbGameWL2021)

print("2021 Schema:")
print(DFmlbGameWL.printSchema())
print("\n2021 DF:")
print(DFmlbGameWL.show(10))


testDFmlbGameWL_for_test_later  = DFmlbGameWL
(trainingDFmlbGameWL, testDFmlbGameWL) = DFmlbGameWL.randomSplit([0.7, 0.3],420)

vis_weight = (trainingDFmlbGameWL.filter('label == 0').count()/trainingDFmlbGameWL.count())
home_weight = (trainingDFmlbGameWL.filter('label == 1').count()/trainingDFmlbGameWL.count())
#print(wiki_weight , au_weight)
trainingDFmlbGameWL = trainingDFmlbGameWL.withColumn("weight", func.when(func.col("label")==1, home_weight).otherwise(vis_weight))

#----DTC----
DTc = DecisionTreeClassifier(featuresCol="featuresVector",labelCol="label",predictionCol="label_predict",weightCol="weight",\
    minInstancesPerNode=10)
DTCModel = DTc.fit(trainingDFmlbGameWL)
dfDTCModelPredic = DTCModel.transform(testDFmlbGameWL)
lablelPrediction = dfDTCModelPredic.select("label","label_predict").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
#print(lablelPrediction.take(5))
metrics = MulticlassMetrics(lablelPrediction)
#stats
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1score = metrics.fMeasure(1.0)
ConfMatrix = metrics.confusionMatrix().toArray().astype(int)
accuracy = (ConfMatrix[0][0] + ConfMatrix[1][1])/(ConfMatrix[1][0] + ConfMatrix[1][1] + ConfMatrix[0][0] + ConfMatrix[0][1])
print('2021 Test Split DTC: Precision:',precision)
print('2021 Test Split DTC: Recall:',recall)
print('2021 Test Split DTC: F1 Score:',f1score)
print('2021 Test Split DTC: Accuracy:',accuracy)
print('2021 Test Split DTC: Confusionn Matrix:',ConfMatrix)
print('\n2021 Test Split Decision Tree Model:\n',DTCModel.toDebugString,"\n")
#print(testDFmlbGameWL.count())


#-----------------------------------------------------now lets see how well this does for the previous year, 2020---------------------------------------------------------
print('\n2020 tested by the 2021 Train Split:\n')
#----read in data----
#local
gamedata2020 = r"C:\Users\jamesbloor2\Desktop\BU\CS777\2020_gamelog_data.csv"
gamedata2020 = sc.textFile(gamedata2020)
#online
#corpus = sc.textFile(sys.argv[1])
gamedata2020 = gamedata2020.map(lambda x: x.split(','))
#count = gamedata.count()
#print(count)
#print(gamedata.take(2))

#----Clean data up----
#3,4,5,7,8,13,17,18,78,80,82,84,90,92,102,104,106,109,112,115,118,121,124,127,130,133,136,139,142,145,148,151,154,157,10,11
#select columns wanted
mlbgamedata = gamedata2020.map(lambda x: (x[3-1], x[4-1]+"V",x[5-1]+"V",x[7-1]+"H",x[8-1]+"H",x[13-1],x[17-1],x[78-1],x[80-1],x[82-1],x[84-1]\
    ,x[90-1]+"V",x[92-1]+"H",x[102-1]+"V",x[104-1]+"H",x[106-1]+"V",x[109-1]+"V",x[112-1]+"V",x[115-1]+"V",x[118-1]+"V",x[121-1]+"V",x[124-1]+"V"\
        ,x[127-1]+"V",x[130-1]+"V",x[133-1]+"H",x[136-1]+"H",x[139-1]+"H",x[142-1]+"H",x[145-1]+"H",x[148-1]+"H",x[151-1]+"H",x[154-1]+"H",x[157-1]+"H",x[10-1],x[11-1]))
#print(mlbgamedata.take(3))

#make sure data is fully filled
mlbgamedataCleaned = mlbgamedata.filter(correctRows)

#how much data was filtered out?
countBeforeClean = mlbgamedata.count()
countAfterClean = mlbgamedataCleaned.count()
print("2020 Count before clean:", countBeforeClean,"\n", "2020 Count after clean:",countAfterClean)

#----make my own columns that I will need----
#1 will be a home win
#0 will be a visitor win 
mlbGameWL = mlbgamedataCleaned.map(lambda x: (1 if (x[34]>x[33]) else 0, (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],\
    x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],x[23],x[24],x[25],x[26],x[27],x[28],x[29],x[30],x[31],\
        x[32])))

#print(mlbGameWL.take(3))

schema = StructType([ 
    StructField("label",IntegerType(),True), 
    StructField("features",ArrayType(StringType(),True))])
DFmlbGameWL2020 = spark.createDataFrame(data=mlbGameWL,schema=schema).cache()
#model = cv.fit(DFmlbGameWL)
DFmlbGameWL = cvmodel.transform(DFmlbGameWL2020)


#----DTC----
dfDTCModelPredic = DTCModel.transform(DFmlbGameWL)
lablelPrediction = dfDTCModelPredic.select("label","label_predict").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
#print(lablelPrediction.take(5))
metrics = MulticlassMetrics(lablelPrediction)
#stats
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1score = metrics.fMeasure(1.0)
ConfMatrix = metrics.confusionMatrix().toArray().astype(int)
accuracy = (ConfMatrix[0][0] + ConfMatrix[1][1])/(ConfMatrix[1][0] + ConfMatrix[1][1] + ConfMatrix[0][0] + ConfMatrix[0][1])
print('2020 DTC: Precision:',precision)
print('2020 DTC: Recall:',recall)
print('2020 DTC: F1 Score:',f1score)
print('2020 DTC: Accuracy:',accuracy)
print('2020 DTC: Confusionn Matrix:',ConfMatrix)
#print('2020 Decision Tree Model:\n',DTCModel.toDebugString)



#---------------------------------------------------now lets see how well this does for the previous year, 2019----------------------------------------------------
print('\n2019 tested by the 2021 Train Split:\n')
#----read in data----
#local
gamedata2019 = r"C:\Users\jamesbloor2\Desktop\BU\CS777\2019_gamelog_data.csv"
gamedata2019 = sc.textFile(gamedata2019)
#online
#corpus = sc.textFile(sys.argv[1])
gamedata2019 = gamedata2019.map(lambda x: x.split(','))
#count = gamedata.count()
#print(count)
#print(gamedata.take(2))

#----Clean data up----
#3,4,5,7,8,13,17,18,78,80,82,84,90,92,102,104,106,109,112,115,118,121,124,127,130,133,136,139,142,145,148,151,154,157,10,11
#select columns wanted
mlbgamedata = gamedata2019.map(lambda x: (x[3-1], x[4-1]+"V",x[5-1]+"V",x[7-1]+"H",x[8-1]+"H",x[13-1],x[17-1],x[78-1],x[80-1],x[82-1],x[84-1]\
    ,x[90-1]+"V",x[92-1]+"H",x[102-1]+"V",x[104-1]+"H",x[106-1]+"V",x[109-1]+"V",x[112-1]+"V",x[115-1]+"V",x[118-1]+"V",x[121-1]+"V",x[124-1]+"V"\
        ,x[127-1]+"V",x[130-1]+"V",x[133-1]+"H",x[136-1]+"H",x[139-1]+"H",x[142-1]+"H",x[145-1]+"H",x[148-1]+"H",x[151-1]+"H",x[154-1]+"H",x[157-1]+"H",x[10-1],x[11-1]))
#print(mlbgamedata.take(3))

#make sure data is fully filled
mlbgamedataCleaned = mlbgamedata.filter(correctRows)

#how much data was filtered out?
countBeforeClean = mlbgamedata.count()
countAfterClean = mlbgamedataCleaned.count()
print("2019 Count before clean:", countBeforeClean,"\n", "2019 Count after clean:",countAfterClean)

#----make my own columns that I will need----
#1 will be a home win
#0 will be a visitor win 
mlbGameWL = mlbgamedataCleaned.map(lambda x: (1 if (x[34]>x[33]) else 0, (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],\
    x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],x[23],x[24],x[25],x[26],x[27],x[28],x[29],x[30],x[31],\
        x[32])))

#print(mlbGameWL.take(3))

schema = StructType([ 
    StructField("label",IntegerType(),True), 
    StructField("features",ArrayType(StringType(),True))])
DFmlbGameWL2019 = spark.createDataFrame(data=mlbGameWL,schema=schema).cache()
#model = cv.fit(DFmlbGameWL)
DFmlbGameWL = cvmodel.transform(DFmlbGameWL2019)



#----DTC----
dfDTCModelPredic = DTCModel.transform(DFmlbGameWL)
lablelPrediction = dfDTCModelPredic.select("label","label_predict").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
#print(lablelPrediction.take(5))
metrics = MulticlassMetrics(lablelPrediction)
#stats
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1score = metrics.fMeasure(1.0)
ConfMatrix = metrics.confusionMatrix().toArray().astype(int)
accuracy = (ConfMatrix[0][0] + ConfMatrix[1][1])/(ConfMatrix[1][0] + ConfMatrix[1][1] + ConfMatrix[0][0] + ConfMatrix[0][1])
print('2019 DTC: Precision:',precision)
print('2019 DTC: Recall:',recall)
print('2019 DTC: F1 Score:',f1score)
print('2019 DTC: Accuracy:',accuracy)
print('2019 DTC: Confusionn Matrix:',ConfMatrix)
#print('2021 Decision Tree Model:\n',DTCModel.toDebugString)



#----------------------------------------------now let's join the 2019 and 2020 data trin and see how it preforms the test on 2021----------------------------------------------
print('\n2019/2020 Train to test on the 2021 DF:\n')
DFmlbGameWL20192020 = unionAll([DFmlbGameWL2019, DFmlbGameWL2020])
print("2019/2020 game count:", DFmlbGameWL20192020.count())
cvmodel = cv.fit(DFmlbGameWL20192020)
DFmlbGameWL = cvmodel.transform(DFmlbGameWL20192020)

print("2019/2020 Schema:")
print(DFmlbGameWL.printSchema())
print("\n2019/2020 DF:")
print(DFmlbGameWL.show(10))

trainingDFmlbGameWL = DFmlbGameWL
testDFmlbGameWL  = testDFmlbGameWL_for_test_later

vis_weight = (trainingDFmlbGameWL.filter('label == 0').count()/trainingDFmlbGameWL.count())
home_weight = (trainingDFmlbGameWL.filter('label == 1').count()/trainingDFmlbGameWL.count())
#print(wiki_weight , au_weight)
trainingDFmlbGameWL = trainingDFmlbGameWL.withColumn("weight", func.when(func.col("label")==1, home_weight).otherwise(vis_weight))

#----DTC----
DTc = DecisionTreeClassifier(featuresCol="featuresVector",labelCol="label",predictionCol="label_predict",weightCol="weight",\
    minInstancesPerNode=10)
DTCModel = DTc.fit(trainingDFmlbGameWL)
dfDTCModelPredic = DTCModel.transform(testDFmlbGameWL)
lablelPrediction = dfDTCModelPredic.select("label","label_predict").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
#print(lablelPrediction.take(5))
metrics = MulticlassMetrics(lablelPrediction)
#stats
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1score = metrics.fMeasure(1.0)
ConfMatrix = metrics.confusionMatrix().toArray().astype(int)
accuracy = (ConfMatrix[0][0] + ConfMatrix[1][1])/(ConfMatrix[1][0] + ConfMatrix[1][1] + ConfMatrix[0][0] + ConfMatrix[0][1])
print('2021 Tested DTC: Precision:',precision)
print('2021 Tested DTC: Recall:',recall)
print('2021 Tested DTC: F1 Score:',f1score)
print('2021 Tested DTC: Accuracy:',accuracy)
print('2021 Tested DTC: Confusionn Matrix:',ConfMatrix)
print('\n2019/2020 Trained Decision Tree Model:')
print(DTCModel.toDebugString)
#print(testDFmlbGameWL.count())



sc.stop()