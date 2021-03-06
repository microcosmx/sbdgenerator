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
import org.apache.spark.mllib.classification.{SVMModel,SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType,StructField,StringType};

import org.apache.spark.SparkException
import org.apache.spark.sql.catalyst.plans.logical.{OneRowRelation, Union}
import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeExec, ReusedExchangeExec, ShuffleExchange}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

import akka.testkit.TestKitBase

class AliyunDataset2 extends FlatSpec with Matchers with BeforeAndAfterAll with TestKitBase {

    implicit lazy val system = ActorSystem()
    implicit val timeout: Timeout = 1.minute

    var fs: org.apache.hadoop.fs.FileSystem = null
    var fs_conf: Configuration = null
    var spark:SparkSession = null
    var sc: org.apache.spark.SparkContext = null
    var sqlContext: org.apache.spark.sql.SQLContext = null

    override def beforeAll() {
        import org.apache.log4j.Logger
        import org.apache.log4j.Level
        Logger.getLogger("org").setLevel(Level.WARN)
        Logger.getLogger("akka").setLevel(Level.WARN)
        Logger.getLogger("parquet.hadoop").setLevel(Level.WARN)

        fs_conf = new org.apache.hadoop.conf.Configuration
        fs = FileSystem.get(fs_conf)

        val conf = new org.apache.spark.SparkConf
        conf.set("spark.master", "local[*]")
        //conf.set("spark.master", "spark://192.168.20.17:7070")
        conf.set("spark.app.name", "Aliyun")
        conf.set("spark.ui.port", "55555")
        conf.set("spark.default.parallelism", "10")
        conf.set("spark.sql.shuffle.partitions", "10")
        conf.set("spark.sql.shuffle.partitions", "1")
        conf.set("spark.sql.autoBroadcastJoinThreshold", "1")
        
        spark = SparkSession.builder
          .config(conf)
          .getOrCreate()

        sc = spark.sparkContext
        sqlContext = spark.sqlContext
        
    }

    override def afterAll() {
        sc.stop()
        fs.close()
        system.shutdown()
    }

    it should "run it" in {
        val cfg = new Config("conf/server.properties")
        
        val jdbc = null
        val ml = MLSample(sc,sqlContext)
        val ss = SStream(sc)
        val env = Env(system, cfg, fs, jdbc, ml, sc, sqlContext)
        

        try{
            import env.sqlContext.implicits._
            
//            val lines = env.sc.textFile("data/mars_tianchi_songs.csv")
//              .map(_.split(",", -1).map(_.trim)).collect
//            val songs = lines
//              .map( line =>
//                mars_tianchi_songs(
//                  song_id = line(0).toString, 
//                  artist_id = line(1).toString, 
//                  publish_time = line(2).toString, 
//                  song_init_plays = line(3).toString, 
//                  Language = line(4).toString, 
//                  Gender = line(5).toString)
//              ).toSeq
//            val songDF = env.sc.parallelize(songs).toDS.as("song")
            
//            implicit val kryoEncoder = Encoders.kryo[mars_tianchi_songs]
            val schema1 = StructType(Array(
                    StructField("song_id", StringType, true),
                    StructField("artist_id", StringType, true),
                    StructField("publish_time", StringType, true),
                    StructField("song_init_plays", StringType, true),
                    StructField("Language", StringType, true),
                    StructField("Gender", StringType, true)
                ))
            
            val lines = spark.read
                .schema(schema1)
                .option("header", "false")
                //.option("inferSchema", "true")
                .csv("data/mars_tianchi_songs.csv")
            //val songDF = lines.as[mars_tianchi_songs]
            val songDF = lines.as[mars_tianchi_songs].as("song")
            songDF.printSchema()
            songDF.show()
 
            val mtua = spark.read.text("data/mars_tianchi_user_actions.csv")
            val mtua_rdd = mtua.as[String].map(_.split(",", -1).map(_.trim)).map { line =>
                mars_tianchi_user_actions(
                  user_id = line(0).toString, 
                  song_id = line(1).toString, 
                  gmt_create = line(2).toString, 
                  action_type = line(3).toString, 
                  Ds = line(4).toString)}
           val uactDF = mtua_rdd.as("uact")
           uactDF.printSchema()
           uactDF.show()
            
            //join
           //action_type
           // 1 for plays
           // 2 for downloads
           // 3 for favors
            val result1 = songDF
                  .filter(_.artist_id == "03c6699ea836decbc5c8fc2dbae7bd3b")
                  //.join(uactDF, songDF("song_id") === uactDF("song_id"))
                  .joinWith(uactDF, $"song.song_id" <=> $"uact.song_id")
                  .filter($"_1.publish_time" >= $"_2.Ds")
                  .filter(_._2.action_type == "1")
                  .groupBy($"_1.artist_id" as "artist_id", $"_2.Ds" as "Ds")
                  .agg(count("_2.action_type") as "Plays", max("_1.song_init_plays") as "initCount",
                      max("_1.publish_time") as "pub_date")
                  .sort("Ds").sort("artist_id")
                  .select("artist_id","Plays","Ds", "initCount","pub_date")

              result1.printSchema
              result1.show
              
              result1.explain()
              
          

//            println("--------IO---------")
//            import java.io._
//            val writer = new PrintWriter(new File("data/result.csv"))
//           
//            var time1 = new Date().getTime
//            println(time1)
//            result1.collect.foreach { 
//               case Row(aid: String,play: String,ds: String,ic: String,pub: String) =>
//                  writer.println(s"$aid,$play,$ds,$ic,$pub")
//               case Row(aid: String,play: Long,ds: String,ic: String,pub: String) =>
//                  val dss = ds.substring(0, 4) + '/' + ds.substring(4, 6) + '/' + ds.substring(6)
//                  val pubs = pub.substring(0, 4) + '/' + pub.substring(4, 6) + '/' + pub.substring(6)
//                  writer.println(s"$aid,$play,$dss,$ic,$pubs")
//               
//               case _ => 
//            }
//            println(new Date().getTime - time1)
//            writer.close()
//            
//            val srcPath = new Path("data/result_1")
//            val dstPath = new Path("data/result_1.txt")
//            result1.rdd.saveAsTextFile("data/result_1")
//            val srcs = FileUtil.stat2Paths(fs.globStatus(srcPath), srcPath)
//            for (src <- srcs) {
//                FileUtil.copyMerge(fs, src,
//                        fs, dstPath, false, fs_conf,null)
//            }
              
              
            //result1.rdd.repartition(1).saveAsTextFile("data/mars_tianchi_artist_plays_predict.csv")
            //result1.rdd.coalesce(1,true).saveAsTextFile("data/mars_tianchi_artist_plays_predict.csv")
            
            //env.ml.LinearRegressionTest(result1)

            
        }
        catch {
            case t: Throwable =>
                t.printStackTrace()
        }
    }


}
