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


class ProcessGen5 extends FlatSpec with Matchers with BeforeAndAfterAll with TestKitBase {
  
    implicit lazy val system = ActorSystem()

    override def beforeAll() {
    }

    override def afterAll() {
        system.shutdown()
    }
    
    def reflectTest() = {
      println("--------reflect----------")
      "reflect success"
    }
    
    

    it should "run it" in {

        try{
            import GA._
            
            var population =  initPopulation(CalFitnessTwo)
            
            println(population)
            
            var smallest = population.head.fitness
            var smallestPlan = population.head
            var temp = 0.0
            for(i <- 0 until MAX_GENERATION){
              population = selectChromosome(population)
              println(s"--------------gen $i----------------")
              println(population.map(x=>x.sequence.toSeq).toSeq)
              //population = CrossOver_Mutation(population, CalFitnessOne)
              population = CrossOver_Mutation(population, CalFitnessTwo)
              temp = population.head.fitness
              if(temp < smallest) {
                smallest = temp
                smallestPlan = population.head
              }
            }
            
            println("--------------result----------------")
            println(s"--transforms: ${smallestPlan.sequence.toSeq}")
            println(s"RMSE = $smallest")
            
            
            //persist
            val result = dataTransformProcess(smallestPlan.sequence)
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
