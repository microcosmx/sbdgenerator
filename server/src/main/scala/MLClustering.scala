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


case class MLClustering(
    spark: SparkSession)
{
  
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
    
    def K_means() = {
        import org.apache.spark.ml.clustering.KMeans

        // Loads data.
        val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
        
        // Trains a k-means model.
        val kmeans = new KMeans().setK(2).setSeed(1L)
        val model = kmeans.fit(dataset)
        
        // Evaluate clustering by computing Within Set Sum of Squared Errors.
        val WSSSE = model.computeCost(dataset)
        println(s"Within Set Sum of Squared Errors = $WSSSE")
        
        // Shows the result.
        println("Cluster Centers: ")
        model.clusterCenters.foreach(println)
    }
    
    def Latent_Dirichlet_allocation() = {
        import org.apache.spark.ml.clustering.LDA
        import org.apache.spark.ml.evaluation.Evaluator

        // Loads data.
        val dataset = spark.read.format("libsvm")
          .load("data/mllib/sample_lda_libsvm_data.txt")
        
        // Trains a LDA model.
        val lda = new LDA().setK(10).setMaxIter(10)
        val model = lda.fit(dataset)
        
        val ll = model.logLikelihood(dataset)
        val lp = model.logPerplexity(dataset)
        println(s"The lower bound on the log likelihood of the entire corpus: $ll")
        println(s"The upper bound bound on perplexity: $lp")
        
        // Describe topics.
        val topics = model.describeTopics(3)
        println("The topics described by their top-weighted terms:")
        topics.show(false)
        
        // Shows the result.
        val transformed = model.transform(dataset)
        transformed.show(false)
    }
    
    def Bisecting_k_means() = {
        import org.apache.spark.ml.clustering.BisectingKMeans

        // Loads data.
        val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
        
        // Trains a bisecting k-means model.
        val bkm = new BisectingKMeans().setK(2).setSeed(1)
        val model = bkm.fit(dataset)
        
        // Evaluate clustering.
        val cost = model.computeCost(dataset)
        println(s"Within Set Sum of Squared Errors = $cost")
        
        // Shows the result.
        println("Cluster Centers: ")
        val centers = model.clusterCenters
        centers.foreach(println)
    }
    
    def Gaussian_Mixture_Model() = {
        import org.apache.spark.ml.clustering.GaussianMixture

        // Loads data
        val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
        
        // Trains Gaussian Mixture Model
        val gmm = new GaussianMixture()
          .setK(2)
        val model = gmm.fit(dataset)
        
        // output parameters of mixture model model
        for (i <- 0 until model.getK) {
          println("weight=%f\nmu=%s\nsigma=\n%s\n" format
            (model.weights(i), model.gaussians(i).mean, model.gaussians(i).cov))
        }
    }
    
    
    
    
    
    
    
    def Collaborative_filtering() = {
        import org.apache.spark.ml.evaluation.RegressionEvaluator
        import org.apache.spark.ml.recommendation.ALS
        
        import spark.sqlContext.implicits._
        
        case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long) extends Serializable
        //implicit val rating = Encoders.kryo[Rating]
        
        def parseRating(str: String) = {
          val fields = str.split("::")
          assert(fields.size == 4)
          //println(Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong))
          (fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
        }
        
        val ratings = spark.read.textFile("data/mllib/als/sample_movielens_ratings.txt")
          .map(parseRating)//(Encoders.javaSerialization[Rating])
          .toDF("userId", "movieId", "rating", "timestamp")
        val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
        
        training.show
        
        // Build the recommendation model using ALS on the training data
        val als = new ALS()
          .setMaxIter(5)
          .setRegParam(0.01)
          .setUserCol("userId")
          .setItemCol("movieId")
          .setRatingCol("rating")
          .setImplicitPrefs(true)
        val model = als.fit(training)
        
        // Evaluate the model by computing the RMSE on the test data
        val predictions = model.transform(test)
        predictions.show
        
        val evaluator = new RegressionEvaluator()
          .setMetricName("rmse")
          .setLabelCol("rating")
          .setPredictionCol("prediction")
        val rmse = evaluator.evaluate(predictions)
        println(s"Root-mean-square error = $rmse")
    }
  
}