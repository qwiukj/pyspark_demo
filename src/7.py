from pyspark.sql import Row,SparkSession
from pyspark.ml.clustering import GaussianMixture, GaussianMixtureModel
from pyspark.ml.linalg import Vectors

def f(x):
    rel = {}
    rel['features'] = Vectors.dense(float(x[1]),float(x[2]),float(x[3]),float(x[4]))
    return rel

spark = SparkSession.builder.getOrCreate()
data = spark.sparkContext.textFile("file:///home/lyw/iris.txt").map(lambda a:a.split(" ")).map(lambda p:Row(**f(p))).toDF()
#data.show(n = 150)
gm = GaussianMixture().setK(3).setPredictionCol("pre").setProbabilityCol("pro")
m = gm.fit(data)
result = m.transform(data)
result.show(150)

for i in range(3):
    print("Component "+str(i)+" : weight is "+str(m.weights[i])+"\n mu vector is "+str( m.gaussiansDF.select('mean').head())+" \n sigma matrix is "+ str(m.gaussiansDF.select('cov').head()))