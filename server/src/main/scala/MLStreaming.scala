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


case class MLStreaming(
    spark: SparkSession)
{
    val sc = spark.sparkContext
  
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
        
    def Streaming_k_means() = {
        import org.apache.spark.mllib.clustering.StreamingKMeans
        import org.apache.spark.mllib.linalg.Vectors
        import org.apache.spark.mllib.regression.LabeledPoint
        import org.apache.spark.streaming.{Seconds, StreamingContext}
        
        val ssc = new StreamingContext(sc, Seconds(10))
        
        val trainingData = ssc.textFileStream("data/mllib/streaming").map(Vectors.parse)
        val testData = ssc.textFileStream("data/mllib/streaming").map(LabeledPoint.parse)
        
        val model = new StreamingKMeans()
          .setK(2)
          .setDecayFactor(1.0)
          .setRandomCenters(1, 0.0)
        
        model.trainOn(trainingData)
        model.predictOnValues(testData.map(lp => (lp.label, lp.features))).print()
        
        ssc.start()
        ssc.awaitTermination()
    }
  
}