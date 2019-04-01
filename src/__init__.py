from pyspark.ml.linalg import Vectors
from pyspark.sql import Row,functions
a ={}
a["au"] = Vectors.dense(1,2,3)
a["b"] = 2
print(Row(**a))