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

class ProcessGen2 extends FlatSpec with Matchers with BeforeAndAfterAll with TestKitBase {

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
        

        try{
            import env.sqlContext.implicits._
            
            val zip = spark.read
                //.schema(schema1)
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("data2/superzip.csv")
            val superzipDS = zip.as("superzip")
            superzipDS.printSchema()
            superzipDS.show()
            
            
            val flds = superzipDS.schema.fields
            val fNames = superzipDS.schema.fieldNames
            println(flds.length)
            println(flds(0).name)
            println(flds(0).dataType.simpleString )
            
            //reflect
            import scala.reflect.runtime.{universe => ru}
            val typeMirror = ru.runtimeMirror(this.getClass.getClassLoader)
            val instanceMirror = typeMirror.reflect(this)
            val members = instanceMirror.symbol.typeSignature.members
            members.map {symbol =>
              val name = symbol.name.toString.trim
              if(name == "reflectTest"){
                instanceMirror.reflectMethod(symbol.asMethod)()
              }
            }
            
            val typemethodMap = Map(
                "int" -> "getInt",
                "long" -> "getLong",
                "double" -> "getDouble",
                "string" -> "getString",
                "date" -> "getDate"
            )
            val transDS = superzipDS
                  .filter(x => {
                      val typeMirror = ru.runtimeMirror(x.getClass.getClassLoader)
                      val instanceMirror = typeMirror.reflect(x)
                      val methodX = ru.typeOf[Row].declaration(ru.newTermName(typemethodMap(flds(0).dataType.simpleString))).asMethod
                      instanceMirror.reflectMethod(methodX)(0).asInstanceOf[Int] > 1100
                      //x.getInt(0) > 1100
                  })
                  .sort(fNames(0))
            
            transDS.printSchema()
            transDS.show()
            
            
            //eval
            println("--------eval-----------")
            import scala.tools.reflect.ToolBox
            val tb = scala.reflect.runtime.currentMirror.mkToolBox()
            val tree = tb.parse("1 to 3 map (_+1)")
            val result = tb.eval(tree)
            println(result)
            
            /*
            //val bValue = sc.broadcast(tb)
            val transDS2 = superzipDS
                  .filter(x => {
                      import scala.tools.reflect.ToolBox
                      val tb = scala.reflect.runtime.currentMirror.mkToolBox()
                      val tree = tb.parse("x.getInt(0) > 1100")
                      val result = tb.eval(tree)
                      result.asInstanceOf[Boolean]
                      //x.getInt(0) > 1100
                  })
                  .sort(fNames(0))
            
            transDS2.printSchema()
            transDS2.show()
            */
            
            
            //eval2
            println("--------eval2-----------")
            /*
            import scala.tools.nsc.interpreter.IMain
            val interpreter = new IMain
            interpreter.interpret("""val x = 6.28
                val eval = math.sin(x) * x""")
            val x = interpreter.valueOfTerm("x")
            println(x)
            import scala.tools.nsc._
            import scala.tools.nsc.interpreter._
            val n=new Interpreter(new Settings())
            n.bind("label", "Int", new Integer(4))
            n.interpret("println(2+label)")
            n.close()
            */
            
            
            //eval3 -- twitter util
            /////https://github.com/twitter/util/
            println("--------eval3-----------")
            
            import com.twitter.util._
            import com.twitter.io.TempFile
            import java.io.{File, FileWriter}
            import org.scalatest.WordSpec
            import org.scalatest.junit.JUnitRunner
            import scala.io.Source
            import scala.language.reflectiveCalls
            import scala.reflect.internal.util.Position
            import scala.tools.nsc.Settings
            import scala.tools.nsc.reporters.{AbstractReporter, Reporter}
            
            println((new Eval).apply[Int]("1 + 1"))
            
            val code_saple = """
              object EvalObj {
                val name:String="CSV"
                def orderString(i: String): String = {
                  i.split(",").mkString(",")
                }
              }
              val records = Array(
                "1,2,3,4,5",
                "9,8,7,6,5"
                )
              records.foreach( i => println(EvalObj.orderString( i )))
            """
            trait EvalObj {
              val name:String
              def orderString(i: String)
            }
            val eval = new Eval // Initializing The Eval without any target location
            //val tsvEval = eval[EvalObj](code_saple)
            //tsvEval.orderString( "2,5,10" )
            
            
            class Ctx {
              val eval = new Eval {
                @volatile var errors: Seq[(String, String)] = Nil
                override lazy val compilerMessageHandler: Option[Reporter] = Some(new AbstractReporter {
                  override val settings: Settings = compilerSettings
                  override def displayPrompt(): Unit = ()
                  override def display(pos: Position, msg: String, severity: this.type#Severity): Unit = {
                    errors = errors :+ ((msg, severity.toString))
                  }
                  override def reset() = {
                    super.reset()
                    errors = Nil
                  }
                })
              }
            }
            val ctx = new Ctx
            import ctx._
            println(eval[Int]("val a = 3; val b = 2; a + b", true))
            
            
            //persist
            import HadoopConversions._
            val writeRDD = sc.makeRDD(Seq(superzipDS.schema.fieldNames.mkString(","))) ++
                              transDS.map(row=>row.mkString(",")).rdd
            env.fs.writeFileRDD(writeRDD, "data2/trans.csv")

            
        }
        catch {
            case t: Throwable =>
                t.printStackTrace()
        }
    }


}
