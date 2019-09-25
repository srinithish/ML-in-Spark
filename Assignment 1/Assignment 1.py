# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:35:30 2019

@author: GELab
"""

from pyspark import SparkConf, SparkContext

import numpy as np
import random


from pyspark.sql import SparkSession



spark = SparkSession.builder.appName("Forest").getOrCreate()



sparkDF = spark.read.csv("./MSD.txt",header=False,sep=",")





#myRDD = spark.sparkContext.parallelize(sparkDF)

"""
Question 1
"""
## number of rows
sparkDF.count()

## top rows
sparkDF.head(40)





