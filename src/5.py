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
df = spark.sql("select * from iris")

print(df.count())
rel = df.rdd.map(lambda  t:str(t[1]+":"+str(t[0]))).collect()
for r in rel:
    print(r)
labeli = StringIndexer().setInputCol("lab").setOutputCol("indexl").fit(df)
fea = VectorIndexer().setInputCol("fea").setOutputCol("indexf").setMaxCategories(4).fit(df)
labelc = IndexToString().setInputCol("prediction").setOutputCol("prel").setLabels(labeli.labels)
train,test = data.randomSplit([0.7,0.3])

from pyspark.ml.classification import DecisionTreeClassifier,DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
c = DecisionTreeClassifier().setLabelCol('indexl').setFeaturesCol('indexf')
pip = Pipeline().setStages([labeli,fea,c,labelc])
m =pip.fit(train)
pre = m.transform(test)
pre.select("prel","lab","fea").show()

evaluatorClassifier = MulticlassClassificationEvaluator().setLabelCol("indexl").setPredictionCol(
    "prediction").setMetricName("accuracy")

accuracy = evaluatorClassifier.evaluate(pre)
print(accuracy)

from pyspark.ml.regression import DecisionTreeRegressionModel,DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
dtRegressor = DecisionTreeRegressor().setLabelCol("indexl").setFeaturesCol("indexf")
pipelineRegressor = Pipeline().setStages([labeli, fea, dtRegressor, labelc])
modelRegressor = pipelineRegressor.fit(train)

predictionsRegressor = modelRegressor.transform(test)

predictionsRegressor.select("prel", "lab", "fea").show(20)

evaluatorRegressor = RegressionEvaluator().setLabelCol("indexl").setPredictionCol("prediction").setMetricName(
    "rmse")

rmse = evaluatorRegressor.evaluate(predictionsRegressor)

print("Root Mean Squared Error (RMSE) on test data = " + str(rmse))