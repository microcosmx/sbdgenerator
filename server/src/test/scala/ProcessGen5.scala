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


class ProcessGen5 extends FlatSpec with Matchers with BeforeAndAfterAll with TestKitBase{
  
    implicit lazy val system = ActorSystem()
    implicit val timeout: Timeout = 1.minute
    
    var fs: org.apache.hadoop.fs.FileSystem = null
    var fs_conf: Configuration = null
    var spark:SparkSession = null
    var sc: org.apache.spark.SparkContext = null
    var sqlContext: org.apache.spark.sql.SQLContext = null
    
    import org.apache.log4j.Logger
    import org.apache.log4j.Level
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    Logger.getLogger("parquet.hadoop").setLevel(Level.WARN)
    lazy val log: Logger = Logger.getLogger(this.getClass)

    override def beforeAll() {
      
        val fs_conf = new org.apache.hadoop.conf.Configuration
        val fs = FileSystem.get(fs_conf)
      
        val conf = new org.apache.spark.SparkConf
        conf.set("spark.master", "local[*]")
        //conf.set("spark.master", "spark://192.168.20.17:7070")
        conf.set("spark.app.name", "genetic")
        //conf.set("spark.ui.port", "55555")
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
        system.shutdown()
    }
    
    def reflectTest() = {
      println("--------reflect----------")
      "reflect success"
    }
    
    

    it should "run it" in {
      
        val cfg = new Config("conf/server.properties")
        
        val jdbc = null
        val ml = MLSample(sc,sqlContext)
        val ss = SStream(sc)
        val env = Env(system, cfg, fs, jdbc, ml, sc, sqlContext)
        
        val trans = Transform(spark)
        val mlgen = MLGenetor(spark)
        val mlreg = MLRegression(spark)

        try{
            
            val ga = GA(spark, env, trans, mlgen, mlreg)
            
            var population = ga.initPopulation(ga.CalFitnessTwo)
            
            println(population)
            
            var smallest = population.head.fitness
            var smallestPlan = population.head
            var temp = 0.0
            for(i <- 0 until ga.MAX_GENERATION){
              population = ga.selectChromosome(population)
              log.info(s"--------------gen $i----------------")
              log.info(population.map(x=>x.sequence.toSeq).toSeq)
              //population = CrossOver_Mutation(population, CalFitnessOne)
              population = ga.CrossOver_Mutation(population, ga.CalFitnessTwo)
              temp = population.head.fitness
              if(temp < smallest) {
                smallest = temp
                smallestPlan = population.head
              }
            }
            
            log.info("--------------result----------------")
            log.info(s"--transforms: ${smallestPlan.sequence.toSeq}")
            log.info(s"RMSE = $smallest")
            
            
            //persist
            val result = ga.dataTransformProcess(smallestPlan.sequence)
            import HadoopConversions._
            val writeRDD = spark.sparkContext.makeRDD(Seq(result.schema.fieldNames.mkString(","))) ++
                              result.rdd.map(row=>row.mkString(","))
            env.fs.writeFileRDD(writeRDD, "data2/trans.csv")
            
            
        }
        catch {
            case t: Throwable =>
                t.printStackTrace()
        }
    }


}
