from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
doc = spark.createDataFrame([("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )],["text"])
w2v = Word2Vec(vectorSize=9,minCount=0,inputCol="text",outputCol="res")
model = w2v.fit(doc)
docs = model.transform(doc)

for row in docs.collect():
    text,vec = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vec)))