from pyspark.ml.feature import HashingTF,IDF,Tokenizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

sen = spark.createDataFrame([(0, "I heard about Spark and I love Spark"),(0, "I wish Java could use case classes"),(1, "Logistic regression models are neat")]).toDF("label","sentence")
tok = Tokenizer(inputCol="sentence",outputCol="words")
words = tok.transform(sen)
htf = HashingTF(inputCol="words",outputCol="f",numFeatures=20)
fea = htf.transform(words)
idf = IDF(inputCol="f",outputCol="feat")
idfs = idf.fit(fea)
data = idfs.transform(fea)
data.select("label","feat").show()