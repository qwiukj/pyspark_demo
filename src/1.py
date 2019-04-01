from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from  pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF,Tokenizer

spark = SparkSession.builder.getOrCreate()
train = spark.createDataFrame([(0,"a b c spark",1.0),
                               (1,"d e f",0.0),
                               (2,"g h i",0.0),
                               (3,'p q spark',1.0),
                               (4,"o r s",0.0),
                               (5,"spark t u",1.0)],
                              ["id", "text", "label"])

tokenizer = Tokenizer(inputCol="text",outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(),outputCol="features")
lr = LogisticRegression(maxIter=10,regParam=0.001)

pipeline = Pipeline(stages=[tokenizer,hashingTF,lr])

model = pipeline.fit(train)

test = spark.createDataFrame([(6,"v w x"),
                              (7,"y z yy"),
                              (8,"zz spark tt"),
                              (9,"aa bb spark"),
                              (10,"bi ni mi spark")],
                             ["id", "text"])

prediction = model.transform(test)
select = prediction.select("id","text","probability","prediction")
for row in select.collect():
    rid,text,prob,pre = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), pre))

from pyspark.ml.feature import StringIndexer
df = spark.createDataFrame([(0,"a"),(1,"b"),(2,"c"),(3,"a"),(4,"a"),(5,"c"),(6,"v")],["id","cate"])
index = StringIndexer(inputCol="cate",outputCol="res")
model = index.fit(df)
ss = model.transform(df)
ss.show()