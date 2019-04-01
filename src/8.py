from pyspark.ml.linalg import Vector,Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel
from pyspark.ml import Pipeline, PipelineModel

def f(x):
    rex = {}
    rex["fea"] = Vectors.dense(float(x[1]),float(x[2]),float(x[3]),float(x[4]))
    rex["lab"] = str(x[5]).replace('"','')
    return rex

from pyspark.sql import SparkSession,SQLContext

spark = SparkSession.builder.getOrCreate()


df = spark.sparkContext.textFile("file:///home/lyw/iris.txt").map(lambda a:a.split(" ")).map(lambda p:Row(**f(p))).toDF()

labeli = StringIndexer().setInputCol("lab").setOutputCol("indexl").fit(df)
feai = VectorIndexer().setInputCol("fea").setOutputCol("indexf").fit(df)

train,test = df.randomSplit([0.7,0.3])
lr = LogisticRegression().setLabelCol("indexl").setFeaturesCol("indexf").setMaxIter(50).setFamily("multinomial")
labelc = IndexToString().setInputCol("prediction").setOutputCol("pl").setLabels(labeli.labels)
pip = Pipeline().setStages([labeli,feai,lr,labelc])

grid = ParamGridBuilder().addGrid(lr.elasticNetParam,[0.2,0.8]).addGrid(lr.regParam,[0.01,0.1,0.5]).build()
cv = CrossValidator().setEstimator(pip).setEvaluator(MulticlassClassificationEvaluator().setLabelCol("indexl").setPredictionCol('prediction')).setEstimatorParamMaps(grid).setNumFolds(5)
cv1 = cv.fit(train)
pre = cv1.transform(test)
res = pre.select("pl",'lab','fea','probability')
res.show(n=150)
eva = MulticlassClassificationEvaluator().setLabelCol("indexl").setPredictionCol("prediction")
a = eva.evaluate(pre)
print(a)


bm = cv1.bestModel
lrModel = bm.stages[2]
print("Coefficients: " + str(lrModel.coefficientMatrix) + "Intercept: "+str(lrModel.interceptVector)+ "numClasses: "+str(lrModel.numClasses)+"numFeatures: "+str(lrModel.numFeatures))
