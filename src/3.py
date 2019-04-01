from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a a a".split(" "))
], ["id", "words"])

cv = CountVectorizer(inputCol="words",outputCol="fea",vocabSize=3,minDF=2.0)
model = cv.fit(df)
res = model.transform(df)
print(res.show())

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([(7,Vectors.dense([0,0,18,1]),1),
(8,Vectors.dense([0,1,12,0]),0),
(9,Vectors.dense([1,0,15,1]),0),
                            ],["id","fea","cli"])
sele = ChiSqSelector(numTopFeatures=2,featuresCol="fea",outputCol="res",labelCol="cli")
res = sele.fit(df).transform(df)
res.show()
