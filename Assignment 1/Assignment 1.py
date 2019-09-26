# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:35:30 2019

@author: GELab
"""

from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import numpy as np
import random
from pyspark.sql.functions import udf



from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("Forest").getOrCreate()

spark.conf.set("spark.driver.maxResultSize", "5g")

sparkDF = spark.read.csv("./Assignment 1/MSD.txt",header=False,sep=",",inferSchema=True)

type(sparkDF)



#myRDD = spark.sparkContext.parallelize(sparkDF)

"""
Question 1
"""
## number of rows
sparkDF.count()

## top rows
sparkDF.show(1)


###columns
sparkDF.columns

# UDF for converting column type from vector to double type
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())



assembler = VectorAssembler(inputCols=['_c78'],outputCol = "hell_Vect")

newDf = assembler.transform(sparkDF)
    
scaler = MinMaxScaler(inputCol="hell_Vect", outputCol="_Scaled")
scalerModel = scaler.fit(newDf)

# rescale each feature to range [min, max].
scaledData = scalerModel.transform(newDf)


def normaliseEntireDf(sparkDf):
    
    origColumns = sparkDf.columns
    
    for i in origColumns:
        # VectorAssembler Transformation - Converting column to vector type
        assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")
    
        # MinMaxScaler Transformation
        scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")
    
        # Pipeline of VectorAssembler and MinMaxScaler
        pipeline = Pipeline(stages=[assembler, scaler])
    
        # Fitting pipeline on dataframe
        sparkDf = pipeline.fit(sparkDf).transform(sparkDf)
        
    return sparkDf

newSparkDf = normaliseEntireDf(sparkDF)

