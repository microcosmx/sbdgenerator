package cdn

import akka.actor._
import akka.io.IO
import akka.pattern._
import akka.routing._
import akka.util.Timeout
import org.apache.hadoop.fs._
import org.apache.avro._
import org.apache.avro.file._
import org.apache.avro.reflect._
import org.apache.hadoop.fs._
import org.apache.hadoop.conf._
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.storage._
import org.apache.spark.streaming._
import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import spray.can.Http
import spray.json._
import DefaultJsonProtocol._
import org.scalatest._
import scala.annotation._
import scala.math.Ordered.orderingToOrdered
import scala.math.Ordering.Implicits._
import scala.util.hashing._
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.mllib.classification.{ SVMModel, SVMWithSGD }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{ StructType, StructField, StringType };

import org.apache.spark.SparkException
import org.apache.spark.sql.catalyst.plans.logical.{ OneRowRelation, Union }
import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.exchange.{ BroadcastExchangeExec, ReusedExchangeExec, ShuffleExchange }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

object CDNSample extends App {

  implicit lazy val system = ActorSystem()
  implicit val timeout: Timeout = 1.minute

  override def main(args: Array[String]) {

    val fs_conf = new org.apache.hadoop.conf.Configuration
    val fs = FileSystem.get(fs_conf)

    val conf = new org.apache.spark.SparkConf
    conf.set("spark.master", "local[*]")
    //conf.set("spark.master", "spark://192.168.20.17:7070")
    conf.set("spark.app.name", "cdn")
    //conf.set("spark.ui.port", "55555")
    conf.set("spark.default.parallelism", "10")
    conf.set("spark.sql.shuffle.partitions", "10")
    conf.set("spark.sql.shuffle.partitions", "1")
    conf.set("spark.sql.autoBroadcastJoinThreshold", "1")

    val sparkSession = SparkSession.builder
      .config(conf)
      .getOrCreate()

    val sc = sparkSession.sparkContext
    val sqlContext = sparkSession.sqlContext

    
    
    
    
    
    import sqlContext.implicits._

    import scala.reflect.internal._
    import scala.reflect.internal.util._

    Seq(1, 2, 3).map(i => (i, i.toString)).toDF("int", "str").createOrReplaceTempView("df")

    val result = sparkSession.sql(
      """
                  |SELECT x.str, COUNT(*)
                  |FROM df x JOIN df y ON x.str = y.str
                  |and x.int > 0 and y.int > 1
                  |GROUP BY x.str
                  |having x.str > 2
                """.stripMargin)

    result.printSchema()
    result.show()
    result.explain()

    val rdd_test = result.queryExecution.executedPlan.execute()
    println(rdd_test.count)
    println(result.queryExecution.executedPlan.treeString)
    
    
    
    val rdd1 = sc.makeRDD(Array(("1","Spark"),("2","Hadoop"),("3","Scala"),("4","Java")),2)
    //建立一个行业薪水的键值对RDD，包含ID和薪水，其中ID为1、2、3、5
    val rdd2 = sc.makeRDD(Array(("1","30K"),("2","15K"),("3","25K"),("5","10K")),2)

    println("//下面做Join操作，预期要得到（1,×）、（2,×）、（3,×）")
    val joinRDD=rdd1.join(rdd2).collect.foreach(println)
    
  }

}