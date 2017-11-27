cluster:
make build
spark-submit --class Main target/scala-2.10/sbdsample-assembly-0.1-SNAPSHOT.jar

test:
sbt test
test-only AliyunDataset

run:
sbt clean compile
sbt run
sbt
show discoveredMainClasses


spark cdh:
"hdfs://quickstart.cloudera/user/hive/warehouse/"
[cloudera@quickstart ~]$ spark-shell --master yarn-client
spark-submit --class org.apache.spark.examples.SparkPi --master yarn \
--deploy-mode cluster SPARK_HOME/examples/lib/spark-examples.jar 10



build:
sbt assembly -Dsbt.log.noformat=true

submit assembly jar:
hadoop fs -mkdir /user/spark/assembly/
hadoop fs -put ./target/scala-2.11/sbdgenerator-assembly-0.1-SNAPSHOT.jar  hdfs://quickstart.cloudera/user/spark/assembly/
hadoop fs -copyFromLocal ./target/scala-2.11/sbdgenerator-assembly-0.1-SNAPSHOT.jar  hdfs://quickstart.cloudera/user/spark/assembly/

sync:
cp -r ./target/scala-2.11/sbdgenerator-assembly-0.1-SNAPSHOT.jar /Users/admin/work/workspace_data/cdh_samples/

yarn:
spark-submit \
--name cdh_demo \
--class cdn.CDNSampleRemote \
--master yarn-client \
--executor-memory 1G \
--total-executor-cores 1 \
/opt/workspace_data/cdh_samples/sbdgenerator-assembly-0.1-SNAPSHOT.jar

spark-submit \
--name cdh_demo \
--class cdn.CDNSampleRemote \
--master yarn-cluster \
--executor-memory 1G \
--total-executor-cores 1 \
/opt/workspace_data/cdh_samples/sbdgenerator-assembly-0.1-SNAPSHOT.jar






