import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Aggregator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.{LinearRegressionModel,RidgeRegressionModel,LassoModel}
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD,LassoWithSGD,LinearRegressionWithSGD}
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

import scala._
import scala.util._

import scala.reflect.runtime.{universe => ru}


case class Transform(
    spark: SparkSession)
{
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
  
    def trans1(data: Dataset[Row], indexs: Seq[Int], stats: MultivariateStatisticalSummary) = {
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      println(s"-----trans1--filter null rows--${featureNames}-----")
      data.filter(row=> {
          indexs.map(idx=>{
              row.get(idx) != null
          }).reduce(_ && _)
      })
    }
    
    def trans2(data: Dataset[Row], indexs: Seq[Int], stats: MultivariateStatisticalSummary) = {
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      val filterColIdx = Random.nextInt(features.length)
      println(s"-----trans2--filter columns--${features(filterColIdx).name}-----")
      data.drop(features(filterColIdx).name)
        .withColumn(features(filterColIdx).name, lit(Random.nextInt(100)))
    }
    
    def trans3(data: Dataset[Row], indexs: Seq[Int], stats: MultivariateStatisticalSummary) = {
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      println(s"-----trans3--math log--${featureNames}-----")
      val typeMap = Map(
            "int" -> "getInt",
            "long" -> "getLong",
            "double" -> "getDouble",
            "string" -> "getString",
            "date" -> "getDate"
        )
      
      val result = data.map(row => {
          val typeMirror = ru.runtimeMirror(row.getClass.getClassLoader)
          val instanceMirror = typeMirror.reflect(row)
          //val idxList = indexs.filter(idx => instanceMirror.reflectMethod(methodX)(idx) != null && features(idx).dataType.simpleString == "double")
          val idxList = (0 to features.length-1).filter(idx => row.get(idx) != null && features(idx).dataType.simpleString == "double")
          val newrowSeq = row.toSeq
          if(idxList.length > 0){ 
              val newrow = newrowSeq.zipWithIndex.map(x=>{
                  if(idxList contains x._2){
                      val methodX = ru.typeOf[Row].declaration(ru.newTermName(typeMap(features(x._2).dataType.simpleString))).asMethod
                      if(stats.variance(x._2)>100)
                        Math.log(instanceMirror.reflectMethod(methodX)(x._2).asInstanceOf[Double])
                      else
                        instanceMirror.reflectMethod(methodX)(x._2).asInstanceOf[Double]
                  }else{
                      x._1
                  }
              })
              Row.fromSeq(newrow)
          }else{
              row
          }
      })(RowEncoder(data.schema))
      //println(s"========${result.isInstanceOf[Dataset[Row]]}=========")
      //val resultDF = spark.createDataFrame(result.rdd, data.schema)
      //println(s"========${resultDF.isInstanceOf[Dataset[Row]]}=========")
      result
    }
    
    def trans4(data: Dataset[Row], indexs: Seq[Int], stats: MultivariateStatisticalSummary) = {
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      println(s"-----trans4--transform null rows--${featureNames}-----")
      val valueMap = Map(
            "int" -> 0,
            "long" -> 0,
            "double" -> 0.0,
            "string" -> "",
            "date" -> "19700101"
        )
      val result = data.map(row=> {
          val newrowSeq = row.toSeq
          val newrow = newrowSeq.zipWithIndex.map(x=>{
              if(indexs contains x._2){
                  if(x._1 == null){
                    valueMap(features(x._2).dataType.simpleString)
                  }else{
                    x._1
                  }
              }else{
                  x._1
              }
          })
          Row.fromSeq(newrow)
      })(RowEncoder(data.schema))
      
      result
    }
    
    def trans5(data: Dataset[Row], indexs: Seq[Int], stats: MultivariateStatisticalSummary) = {
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      println(s"-----trans5--cube--${featureNames}-----")
      if(featureNames.length > 0){
          val featuresLeft = features.filterNot(x=>featureNames contains x.name)
          val aggCols = featuresLeft.map(x=>{
                if(x.dataType.simpleString == "string"){
                  count(x.name) as x.name
                }else if(x.dataType.simpleString == "int"){
                  min(x.name) as x.name
                }else if(x.dataType.simpleString == "long"){
                  max(x.name) as x.name
                }else if(x.dataType.simpleString == "int"){
                  sum(x.name) as x.name
                }else{
                  count(x.name) as x.name
                }
            })
          data.groupBy(featureNames.head, featureNames.tail:_*)
            .agg(aggCols.head, aggCols.tail:_*)
            //.min(featureNamesLeft:_*)
      }else{
          data.sort(features.head.name, features.tail.map(_.name):_*)
      }
      //skip this step
      data.sort(features.head.name, features.tail.map(_.name):_*)
    }
    
}