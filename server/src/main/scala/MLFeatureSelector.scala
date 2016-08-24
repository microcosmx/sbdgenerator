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


case class MLFeatureSelector(
    spark: SparkSession)
{
  
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
    
    def VectorSlicer() = {
        import java.util.Arrays

        import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
        import org.apache.spark.ml.feature.VectorSlicer
        import org.apache.spark.ml.linalg.Vectors
        import org.apache.spark.sql.Row
        import org.apache.spark.sql.types.StructType
        
        val data = Arrays.asList(Row(Vectors.dense(-2.0, 2.3, 0.0)))
        
        val defaultAttr = NumericAttribute.defaultAttr
        val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
        val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])
        
        val dataset = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))
        
        val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")
        
        slicer.setIndices(Array(1))
        //slicer.setNames(Array("f3"))
        // or slicer.setIndices(Array(1, 2)), or slicer.setNames(Array("f2", "f3"))
        
        val output = slicer.transform(dataset)
        println(output.select("userFeatures", "features").first())
        output.show()
    }
    
    def RFormula() = {
        import org.apache.spark.ml.feature.RFormula

        val dataset = spark.createDataFrame(Seq(
          (7, "US", 18, 1.0),
          (8, "CA", 12, 0.0),
          (9, "NZ", 15, 0.0)
        )).toDF("id", "country", "hour", "clicked")
        val formula = new RFormula()
          .setFormula("clicked ~ country + hour")
          .setFeaturesCol("features")
          .setLabelCol("label")
        val output = formula.fit(dataset).transform(dataset)
        output.select("features", "label").show()
    }
    
    def ChiSqSelector() = {
        import org.apache.spark.ml.feature.ChiSqSelector
        import org.apache.spark.ml.linalg.Vectors
        
        val data = Seq(
          (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
          (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
          (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
        )
        
        val df = spark.createDataset(data).toDF("id", "features", "clicked")
        
        val selector = new ChiSqSelector()
          .setNumTopFeatures(2)
          .setFeaturesCol("features")
          .setLabelCol("clicked")
          .setOutputCol("selectedFeatures")
        
        val result = selector.fit(df).transform(df)
        result.show()
    }
    
}