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
import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.streaming._
import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import spray.can.Http
import spray.json._

import scala._
import scala.util._

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Aggregator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute, NumericAttribute}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tree._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row


object Utils {
	  
    def makeDF(sc: SparkContext, sqlContext: org.apache.spark.sql.hive.HiveContext, path: String) = {
          println(path)

          val lines = sc.textFile(s"data/$path").map(_.split(",", -1).map(_.trim)).collect
          
          val schema = StructType(lines.head.map(_.split(":").toSeq).map {
              case Seq(name, "string")   => StructField(name, StringType)
              case Seq(name, "integer")  => StructField(name, IntegerType)
              case Seq(name, "double")   => StructField(name, DoubleType)
              case Seq(name, "boolean")  => StructField(name, BooleanType)
          })

          val rows = lines.tail.map(cols => {
              require(cols.size == schema.fields.size)
              Row.fromSeq(cols zip schema map {
                  case ("", StructField(_, _, true, _)) => null
                  case (col, StructField(_, StringType,  _, _)) => col
                  case (col, StructField(_, IntegerType, _, _)) => col.toInt
                  case (col, StructField(_, DoubleType,  _, _)) => col.toDouble
                  case (col, StructField(_, BooleanType, _, _)) => col.toBoolean
              })
          })

          lazy val df = sqlContext.createDataFrame(sc.parallelize(rows), schema)
          df
      }
    
   
    def preprocessML(dataset: Dataset[Row], preferredTargets:Seq[String]=Seq(), preferredFeatures:Seq[String]=Seq()) = {
        import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
        import scala.reflect.runtime.{universe => ru}
        
        var fs = dataset.schema.fields
        var fsNames = fs.map(_.name).toSeq
        var dataset1 = dataset
        fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="string").map(x=>{
             val indexer = new StringIndexer()
              .setInputCol(x._1.name)
              .setOutputCol(x._1.name+"_index")
            val indexed = indexer.fit(dataset1).transform(dataset1)
            dataset1 = indexed.drop(x._1.name).withColumnRenamed(x._1.name+"_index", x._1.name)
        })
        fs = dataset1.schema.fields
        fsNames = fs.map(_.name).toSeq
        
        var targets = preferredTargets.intersect(fsNames)
        var objects = preferredFeatures.intersect(fsNames)
        objects = objects.diff(targets)
        val cols = targets++objects
        
        println(targets)
        println(objects)
        
        dataset1 = dataset1.select(cols.head, cols.tail:_*)
        fs = dataset1.schema.fields
        fsNames = fs.map(_.name).toSeq
        val tRand = Random.nextInt(targets.length)
        val oRand = Random.nextInt(objects.length)
        
        val datasetRDD = dataset1.rdd.map { row => 
              val typeMirror = ru.runtimeMirror(row.getClass.getClassLoader)
              val instanceMirror = typeMirror.reflect(row)
              val fsValue = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="double").map(x => {
                  val thevalue = row.get(x._2)
                  (x._1.name, if(thevalue==null) 0.0 else thevalue.asInstanceOf[Double])
              }).toMap
              val fsValueInt = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="int").map(x => {
                  val thevalue = row.get(x._2)
                  (x._1.name, if(thevalue==null) 0.0 else thevalue.asInstanceOf[Int].toDouble)
              }).toMap
              val fsValueLong = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="long").map(x => {
                  val thevalue = row.get(x._2)
                  (x._1.name, if(thevalue==null) 0.0 else thevalue.asInstanceOf[Long].toDouble)
              }).toMap
              
              val vectors = fsValue ++ fsValueInt ++ fsValueLong
              val tgs = targets(tRand)
              val obs = objects.take(1+oRand)
              
              val tgsVal = vectors(tgs)
              val obsVal = vectors.filterKeys { x => obs.contains(x) }.map(_._2).toSeq
              
              val label = tgsVal
              val fvector = obsVal
              //val features = Vectors.sparse(fvector.size, Array.range(0, fvector.size), fvector.toArray)
              val features = Vectors.dense(fvector.toArray)
              LabeledPoint(
                label, features
              )
         }
        
        datasetRDD
    }
    
}

