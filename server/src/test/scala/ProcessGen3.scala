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

class ProcessGen3 extends FlatSpec with Matchers with BeforeAndAfterAll with TestKitBase {

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
        sc.stop()
        fs.close()
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
            import scala.util._
            import env.sqlContext.implicits._
            
            import scala.reflect.runtime.{universe => ru}
            val typeMirror = ru.runtimeMirror(trans.getClass.getClassLoader)
            val instanceMirror = typeMirror.reflect(trans)
            val members = instanceMirror.symbol.typeSignature.members
            val transMems = members.filter(_.name.decoded.startsWith("trans"))
            val transNames = transMems.map(_.name.decoded).toSeq.sortBy(x=>x)
            println(transNames)
            
            val zip = spark.read
                //.schema(schema1)
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("data2/superzip.csv")
            val superzipDS = zip.as("superzip")
            
            var transDS = superzipDS//.filter(x => {x.getInt(0) > 1100}).sort(features(0).name)
            
            transDS.printSchema()
            transDS.show()
            
            val features = superzipDS.schema.fields
            val featureNames = features.map(_.name).toSeq
            
            val actions = spark.read
                //.schema(schema1)
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("data2/action.csv")
            val actionsDS = actions.as("actions")
            actionsDS.show
            
            val actionsList = actionsDS.collect.map { row => 
                val target1 = row.getAs[String]("target")
                val action1 = row.getAs[String]("action")
                val object1 = row.getAs[String]("object")
                if(target1!="all" && action1!="related_to" && object1!="any"){
                  ("level1", (target1.split(" ").toSeq, action1.split(" ").toSeq, object1.split(" ").toSeq))
                }else{
                  ("level3", (
                      if(target1.contains("all")) Seq()/*featureNames.toSeq*/ else target1.split(" ").toSeq,
                      action1.split(" ").toSeq, 
                      if(object1.contains("any")) Seq()/*featureNames.toSeq*/ else object1.split(" ").toSeq
                  ))
                }
            }
            val actionLevel1 = actionsList.filter(_._1 == "level1").map(_._2).toSeq
            val actionLevel3 = actionsList.filter(_._1 == "level3").map(_._2).toSeq
            
            actionLevel1.foreach { x=>
                if(x._2.head == "filter"){
                    transDS = transDS.filter(s"${x._1.head} != ${x._3.head}") 
                }else{
                    //TODO
                }
            }
            transDS.show
            
            val length = features.length * transNames.length
            val sequence = Array.fill(length)(0).map(x => x ^ Random.nextInt(2))
            
            var handler = Seq[Tuple2[String, Seq[Int]]]()
            
            for(x <- transNames.zipWithIndex){
              var transIdx = Seq[Int]()
              for(y <- 0 to features.length-1){
                if(sequence(x._2*10 + y) > 0){
                  transIdx :+= y
                }
              }
              handler :+= (x._1, transIdx)
            }
            
            handler.foreach(x => {
              val methodx = ru.typeOf[Transform].declaration(ru.newTermName(x._1)).asMethod
              transDS = instanceMirror.reflectMethod(methodx)(transDS, x._2).asInstanceOf[Dataset[Row]]
            })
            
            transDS.printSchema()
            transDS.show()
            
            val targets_ = actionLevel3.flatMap(_._1).intersect(featureNames).toSeq
            val objects_ = actionLevel3.flatMap(_._3).intersect(featureNames).diff(targets_).toSeq
            val actions_ = actionLevel3.flatMap(_._2).toSeq
            
            val targets_left = featureNames.diff(targets_)
            val objects_left = featureNames.diff(objects_)
            
            val actionLevel3Process = Seq(
                ((targets_, actions_, objects_), 4),
                ((targets_, actions_, objects_left), 3),
                ((targets_left.diff(objects_), actions_, objects_), 2),
                ((targets_left, actions_, objects_left), 1)
            )
            
            val result = actionLevel3Process.map(actps=>{
                val result1 = mlgen.decisionTreeMSE(transDS, actps._1._1, actps._1._3)
                val result2 = mlgen.decisionPipline(transDS, actps._1._1, actps._1._3)
                val result3 = mlgen.decision_Multilayer_perceptron_classifier(transDS, actps._1._1, actps._1._3)
                val result4 = mlreg.decision_randomforest(transDS, actps._1._1, actps._1._3)
                val result5 = mlreg.decision_Gradient_boosted_tree(transDS, actps._1._1, actps._1._3)
                val minresult = Seq(result1, result2, result3, result4, result5).sortBy(x => x).head
                println(minresult, result1, result2, result3, result4, result5)
                val retVal = if(minresult < 0.2) minresult else -1.0
                (retVal, actps._2)
            }).toSeq
            
            val validResult = result.filter(_._1 < 0)
            println(validResult)
            
            val targetValue = validResult.map(_._1).sum / math.pow(validResult.map(_._2).sum, 1.5)
            println(s"-------target value is-------${targetValue}----------")
            
            
            //persist
            //transDS = transDS.limit(100)
            import HadoopConversions._
            
            val contentHeader = ("transform" +: transDS.schema.fieldNames).mkString(",")
            val content = sequence.sliding(10, 10).zipWithIndex.map { seq10 => 
                (transNames(seq10._2) +: seq10._1.map(_.toString)).mkString(",")
            }.toSeq
            env.fs.saveAsTextFile("data2/process.csv", contentHeader +: content)
            
            val writeRDD = sc.makeRDD(Seq(transDS.schema.fieldNames.mkString(","))) ++
                              transDS.map(row=>row.mkString(",")).rdd
            env.fs.writeFileRDD(writeRDD, "data2/trans.csv")
            
            
            //Rscript
            import scala.sys.process._
            val command = Seq(
                "Rscript",
                "data2/plot.R")
            println(s"R command: ${command.mkString(" ")}")
            val exitCode = blocking { command.! }
            //require(exitCode == 0, s"R exited with code $exitCode")

            
        }
        catch {
            case t: Throwable =>
                t.printStackTrace()
        }
    }


}
