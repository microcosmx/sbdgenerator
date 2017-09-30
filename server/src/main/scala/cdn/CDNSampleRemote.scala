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

object CDNSampleRemote extends App {

  implicit lazy val system = ActorSystem()
  implicit val timeout: Timeout = 1.minute

  override def main(args: Array[String]) {

    val fs_conf = new org.apache.hadoop.conf.Configuration
    val fs = FileSystem.get(fs_conf)

    val conf = new org.apache.spark.SparkConf
    conf.set("spark.master", "yarn-client")
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

    import org.apache.hadoop.mapreduce.Job
    import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
    import org.apache.avro.generic.GenericRecord
    import org.apache.parquet.hadoop.ParquetInputFormat
    import org.apache.parquet.avro.AvroReadSupport
    import org.apache.spark.rdd.RDD

    def rddFromParquetHdfsFile(path: String): RDD[GenericRecord] = {
      val job = new Job()
      FileInputFormat.setInputPaths(job, path)
      ParquetInputFormat.setReadSupportClass(job,
        classOf[AvroReadSupport[GenericRecord]])
      return sc.newAPIHadoopRDD(job.getConfiguration,
        classOf[ParquetInputFormat[GenericRecord]],
        classOf[Void],
        classOf[GenericRecord]).map(x => x._2)
    }

    val warehouse = "hdfs://quickstart.cloudera/user/hive/warehouse/"
    val order_items = rddFromParquetHdfsFile(warehouse + "order_items");
    val products = rddFromParquetHdfsFile(warehouse + "products");
    
    
    val orders = order_items.map { x => (
        x.get("order_item_product_id").toString(),
        (x.get("order_item_order_id"), x.get("order_item_quantity")))
    }.join(
      products.map { x => (
        x.get("product_id").toString(),
        (x.get("product_name")))
      }
    ).map(x => (
        scala.Int.unbox(x._2._1._1), // order_id
        (
            scala.Int.unbox(x._2._1._2), // quantity
            x._2._2.toString // product_name
        )
    )).groupByKey()
    
    
    val cooccurrences = orders.map(order =>
      (
        order._1,
        order._2.toList.combinations(2).map(order_pair =>
            (
                if (order_pair(0)._2 < order_pair(1)._2)
                    (order_pair(0)._2, order_pair(1)._2)
                else
                    (order_pair(1)._2, order_pair(0)._2),
                order_pair(0)._1 * order_pair(1)._1
            )
        )
      )
    )
    val combos = cooccurrences.flatMap(x => x._2).reduceByKey((a, b) => a + b)
    val mostCommon = combos.map(x => (x._2, x._1)).sortByKey(false).take(10)

  }

}