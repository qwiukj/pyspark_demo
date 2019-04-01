from pyspark.sql import Row,functions
from pyspark.ml.linalg import Vectors,Vector
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer,HashingTF,Tokenizer
from pyspark.ml.classification import *
def f(x):
    rex = {}
    rex["fea"] = Vectors.dense(float(x[1]),float(x[2]),float(x[3]),float(x[4]))
    rex["lab"] = str(x[5]).replace('"','')
    #print(rex)
    #print(Row(**rex))
    return rex

from pyspark import SparkContext
from pyspark.sql import SparkSession,SQLContext

spark = SparkSession.builder.getOrCreate()


data = spark.sparkContext.textFile("file:///home/lyw/iris.txt").map(lambda a:a.split(" ")).map(lambda p:Row(**f(p))).toDF()
#data.show(n = 150)
data.createOrReplaceTempView("iris")
df = spark.sql("select * from iris")
#df = spark.sql("select * from iris where lab != 'setosa'")
print(df.count())
rel = df.rdd.map(lambda  t:str(t[1]+":"+str(t[0]))).collect()
# for r in rel:
#     print(r)
labeli = StringIndexer().setInputCol("lab").setOutputCol("indexl").fit(df)
feai = VectorIndexer().setInputCol("fea").setOutputCol("indexf").fit(df)

train,test = df.randomSplit([0.7,0.3])
lr = LogisticRegression().setLabelCol("indexl").setFeaturesCol("indexf").setMaxIter(100000).setRegParam(0.1).setElasticNetParam(0.9).setFamily("multinomial")
labelc = IndexToString().setInputCol("prediction").setOutputCol("pl").setLabels(labeli.labels)
pip = Pipeline().setStages([labeli,feai,lr,labelc])
m = pip.fit(train)
p = m.transform(test)
prel = p.select("pl","lab","fea","probability").collect()
for item in prel:
    print(str(item['lab'])+','+str(item['fea'])+',predictedLabel : '+str(item['pl']))

eva = MulticlassClassificationEvaluator().setLabelCol("indexl").setPredictionCol("prediction")
a = eva.evaluate(p)
print(a)

"""
lrmodel = m.stages[2]
summ = lrmodel.summary
#损失函数变化
obj = summ.objectiveHistory
print("======================================")
for i in obj:
    print(i)
print("======================================")
#Roc面积
print(summ.areaUnderROC)
print("--------------------------------------")
f = summ.fMeasureByThreshold
#最大的F值
maxf = f.select(functions.max("F-Measure")).head()[0]
print(maxf)
#最有参数
bestThreshold = f.where(f["F-Measure"]== maxf).select("threshold").head()[0]
print(bestThreshold)
 """