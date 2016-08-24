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


case class MLFeatureExtractor(
    spark: SparkSession)
{
  
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
    
    def TF_IDF() = {
        import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

        val sentenceData = spark.createDataFrame(Seq(
          (0, "Hi I heard about Spark"),
          (0, "I wish Java could use case classes"),
          (1, "Logistic regression models are neat")
        )).toDF("label", "sentence")
        
        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        val wordsData = tokenizer.transform(sentenceData)
        val hashingTF = new HashingTF()
          .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
        val featurizedData = hashingTF.transform(wordsData)
        // alternatively, CountVectorizer can also be used to get term frequency vectors
        
        val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
        val idfModel = idf.fit(featurizedData)
        val rescaledData = idfModel.transform(featurizedData)
        rescaledData.select("features", "label").take(3).foreach(println)
        
    }
    
    
    def Word2Vec() = {
        import org.apache.spark.ml.feature.Word2Vec

        // Input data: Each row is a bag of words from a sentence or document.
        val documentDF = spark.createDataFrame(Seq(
          "Hi I heard about Spark".split(" "),
          "I wish Java could use case classes".split(" "),
          "Logistic regression models are neat".split(" ")
        ).map(Tuple1.apply)).toDF("text")
        
        // Learn a mapping from words to Vectors.
        val word2Vec = new Word2Vec()
          .setInputCol("text")
          .setOutputCol("result")
          .setVectorSize(3)
          .setMinCount(0)
        val model = word2Vec.fit(documentDF)
        val result = model.transform(documentDF)
        result.select("result").take(3).foreach(println)
        
    }
    
    def CountVectorizer() = {
        import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

        val df = spark.createDataFrame(Seq(
          (0, Array("a", "b", "c")),
          (1, Array("a", "b", "b", "c", "a"))
        )).toDF("id", "words")
        
        // fit a CountVectorizerModel from the corpus
        val cvModel: CountVectorizerModel = new CountVectorizer()
          .setInputCol("words")
          .setOutputCol("features")
          .setVocabSize(3)
          .setMinDF(2)
          .fit(df)
        
        // alternatively, define CountVectorizerModel with a-priori vocabulary
        val cvm = new CountVectorizerModel(Array("a", "b", "c"))
          .setInputCol("words")
          .setOutputCol("features")
        
        cvModel.transform(df).select("features").show()
    }
  
}