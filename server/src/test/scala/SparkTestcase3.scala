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
//import org.apache.spark._
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
import org.apache.spark.sql.types.{StructType,StructField,StringType}
import akka.testkit.TestKitBase
import org.apache.spark.repl._


import scala.Option.option2Iterable
import scala.collection.mutable.ListBuffer
import scala.tools.nsc.util.FailedInterrupt
import scala.tools.refactoring.Refactoring
import scala.tools.refactoring.common.Change
import scala.tools.refactoring.common.NewFileChange
import scala.tools.refactoring.common.TextChange
import scala.tools.refactoring.util.CompilerProvider
import scala.tools.refactoring.common.InteractiveScalaCompiler
import scala.tools.refactoring.common.Selections
import language.{ postfixOps, reflectiveCalls }
import scala.tools.refactoring.common.NewFileChange
import scala.tools.refactoring.common.RenameSourceFileChange
import scala.tools.refactoring.implementations.Rename
import scala.tools.refactoring.common.TracingImpl
import scala.tools.refactoring.util.UniqueNames

import scala.tools.refactoring._
import scala.tools.refactoring.util._
import scala.tools.refactoring.analysis._
import scala.tools.refactoring.common._
import scala.tools.refactoring.implementations._
import scala.tools.refactoring.sourcegen._
import scala.tools.refactoring.transformation._

import org.apache.spark.util._

import org.apache.spark.sql._
import org.apache.spark.sql.execution.SparkPlanInfo
import org.apache.spark.sql.execution.ui.SparkPlanGraph
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.util.{AccumulatorContext, JsonProtocol}
            
            

class SparkTestcase3 extends FlatSpec with Matchers with BeforeAndAfterAll with TestKitBase {

    implicit lazy val system = ActorSystem()
    implicit val timeout: Timeout = 1.minute

    var fs: org.apache.hadoop.fs.FileSystem = null
    var fs_conf: Configuration = null
    var sparkSession: SparkSession = null
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
        
        sparkSession = SparkSession.builder
          .config(conf)
          .getOrCreate()

        sc = sparkSession.sparkContext
        sqlContext = sparkSession.sqlContext
        

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
            
            //Row("1", 1) :: Row("2", 1) :: Row("3", 1) :: Nil)
            
            
            import org.apache.spark.sql.internal.SQLConf
            import org.apache.spark.sql.types._
            import org.apache.spark.storage.StorageLevel.MEMORY_ONLY
            
            sc.parallelize(1 to 10).map(i => (i, i.toString))
              .toDF().createOrReplaceTempView("sizeTst")
            sparkSession.catalog.cacheTable("sizeTst")
            val stats = sparkSession.table("sizeTst").queryExecution.analyzed.statistics
            println(stats.sizeInBytes)
            println(stats.toString)
            
        }
        catch {
            case t: Throwable =>
                t.printStackTrace()
        }
    }
    


}
