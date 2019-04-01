from pyspark.sql import SparkSession
from operator import add
import os
if __name__ == "__main__":
    #os.system("python --version")
    #os.system("which python")
    spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

    lines = spark.read.text("hdfs://cdh2:8020/test/123.txt").rdd.map(lambda r: r[0])
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(lambda x,y:x+y)
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))

    from pyspark.sql import HiveContext, SQLContext
    from pyspark import  SparkContext,SparkConf

    conf = SparkConf()
    conf.set("hive.metastore.uris","thrift://cdh1:9083")
    conf.set("hive.metastore.warehouse.dir","hdfs://cdh1:9000/user/hive/warehouse")
    conf.set("spark.sql.warehouse.dir","hdfs://cdh1:9000/user/hive/warehouse")
    sc = SparkContext.getOrCreate(conf)

    hive_context = HiveContext(sc)
    print("=====")
    hive_context.sql('show databases')
    hive_context.sql('use test')
    hive_context.sql("""create table ghl (id int ,name string,age int)""")




    spark.stop()