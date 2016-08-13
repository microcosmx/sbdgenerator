import java.lang._
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

import scala.reflect.runtime.{universe => ru}


case class Transform(
    spark: SparkSession)
{
  
    def trans1(data: Dataset[Row], indexs: Seq[Int]) = {
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      println(s"-----trans1----${featureNames}-----")
      data.sort(featureNames(0), featureNames.tail:_*)
    }
    
    def trans2(data: Dataset[Row], indexs: Seq[Int]) = {
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      println(s"-----trans2----${featureNames}-----")
      data.filter(x=> {
          val typeMirror = ru.runtimeMirror(x.getClass.getClassLoader)
          val instanceMirror = typeMirror.reflect(x)
          val methodX = ru.typeOf[Row].declaration(ru.newTermName("get")).asMethod
          indexs.map(idx=>{
              instanceMirror.reflectMethod(methodX)(idx) != null
          }).reduce(_ && _)
      })
    }
    
    def trans3(data: Dataset[Row], indexs: Seq[Int]) = {
      data
    }
    
    def trans4(data: Dataset[Row], indexs: Seq[Int]) = {
      data
    }
    
    def trans5(data: Dataset[Row], indexs: Seq[Int]) = {
      data
    }
    
}