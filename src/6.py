from pyspark.ml.linalg import Vectors,Vector
from pyspark.sql import Row,SparkSession
from pyspark.ml import  Pipeline
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer

spark = SparkSession.builder.getOrCreate()

def f(x):
    rex = {}
    rex["fea"] = Vectors.dense(float(x[1]),float(x[2]),float(x[3]),float(x[4]))
    rex["lab"] = str(x[5]).replace('"','')
    #print(rex)
    #print(Row(**rex))
    return rex

data = spark.sparkContext.textFile("file:///home/lyw/iris.txt").map(lambda a:a.split(" ")).map(lambda p:Row(**f(p))).toDF()
#data.show(n = 150)
data.createOrReplaceTempView("iris")
df = spark.sql('select fea from iris')
from pyspark.ml.clustering import KMeans,KMeansModel

km = KMeans().setK(3).setFeaturesCol('fea').setPredictionCol("pre").fit(df)
res = km.transform(df)
df2 = spark.sql('select * from iris')
res.join(df2,on='fea').show(n=150)
kk = km.clusterCenters()
for k in kk:
    print(k)
