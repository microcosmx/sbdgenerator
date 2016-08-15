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
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
  
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
      val features = data.schema.fields
      val featureNames = indexs.map(x=>features(x).name)
      println(s"-----trans3----${featureNames}-----")
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
          val methodX = ru.typeOf[Row].declaration(ru.newTermName("get")).asMethod
          //val idxList = indexs.filter(idx => instanceMirror.reflectMethod(methodX)(idx) != null && features(idx).dataType.simpleString == "double")
          val idxList = (0 to features.length-1).filter(idx => instanceMirror.reflectMethod(methodX)(idx) != null && features(idx).dataType.simpleString == "double")
          val newrowSeq = row.toSeq
          if(idxList.length > 0){ 
              val newrow = newrowSeq.zipWithIndex.map(x=>{
                  if(idxList contains x._2){
                      val methodX = ru.typeOf[Row].declaration(ru.newTermName(typeMap(features(x._2).dataType.simpleString))).asMethod
                      Math.log(instanceMirror.reflectMethod(methodX)(x._2).asInstanceOf[Double])
                  }else{
                      x._1
                  }
              })
              Row.fromSeq(newrow)
              row
          }else{
              row
          }
      })(RowEncoder(data.schema))
      //println(s"========${result.isInstanceOf[Dataset[Row]]}=========")
      //val resultDF = spark.createDataFrame(result.rdd, data.schema)
      //println(s"========${resultDF.isInstanceOf[Dataset[Row]]}=========")
      result
    }
    
    def trans4(data: Dataset[Row], indexs: Seq[Int]) = {
      data
    }
    
    def trans5(data: Dataset[Row], indexs: Seq[Int]) = {
      data
    }
    
}