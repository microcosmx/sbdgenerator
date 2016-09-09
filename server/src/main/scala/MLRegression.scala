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


case class MLRegression(
    spark: SparkSession)
{
  
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
    
    def linear() = {
        import org.apache.spark.ml.regression.LinearRegression

        // Load training data
        val training = spark.read.format("libsvm")
          .load("data/mllib/sample_linear_regression_data_2.txt")
        
        val lr = new LinearRegression()
          .setMaxIter(10)
          .setRegParam(0.3)
          .setElasticNetParam(0.8)
        
        // Fit the model
        val lrModel = lr.fit(training)
        
        // Print the coefficients and intercept for linear regression
        println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
        
        // Summarize the model over the training set and print out some metrics
        val trainingSummary = lrModel.summary
        println(s"numIterations: ${trainingSummary.totalIterations}")
        println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
        trainingSummary.residuals.show()
        println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
        println(s"r2: ${trainingSummary.r2}")
        
        val predictions = lrModel.transform(training)
        // Select example rows to display.
        predictions.show
    }
    
    def GeneralizedLinearRegression() = {
        import org.apache.spark.ml.regression.GeneralizedLinearRegression

        // Load training data
        val dataset = spark.read.format("libsvm")
          .load("data/mllib/sample_linear_regression_data_2.txt")
        
        val glr = new GeneralizedLinearRegression()
          .setFamily("gaussian")
          .setLink("identity")
          .setMaxIter(10)
          .setRegParam(0.3)
        
        // Fit the model
        val model = glr.fit(dataset)
        
        // Print the coefficients and intercept for generalized linear regression model
        println(s"Coefficients: ${model.coefficients}")
        println(s"Intercept: ${model.intercept}")
        
        // Summarize the model over the training set and print out some metrics
        val summary = model.summary
        println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
        println(s"T Values: ${summary.tValues.mkString(",")}")
        println(s"P Values: ${summary.pValues.mkString(",")}")
        println(s"Dispersion: ${summary.dispersion}")
        println(s"Null Deviance: ${summary.nullDeviance}")
        println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
        println(s"Deviance: ${summary.deviance}")
        println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
        println(s"AIC: ${summary.aic}")
        println("Deviance Residuals: ")
        summary.residuals().show()
        
        val predictions = model.transform(dataset)
        // Select example rows to display.
        predictions.show
    }
    
    def dtree() = {
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.evaluation.RegressionEvaluator
        import org.apache.spark.ml.feature.VectorIndexer
        import org.apache.spark.ml.regression.DecisionTreeRegressionModel
        import org.apache.spark.ml.regression.DecisionTreeRegressor
        
        // Load the data stored in LIBSVM format as a DataFrame.
        val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        // Automatically identify categorical features, and index them.
        // Here, we treat features with > 4 distinct values as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(data)
        
        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        
        // Train a DecisionTree model.
        val dt = new DecisionTreeRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
        
        // Chain indexer and tree in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, dt))
        
        // Train model. This also runs the indexer.
        val model = pipeline.fit(trainingData)
        
        // Make predictions.
        val predictions = model.transform(testData)
        
        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)
        
        val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
        println("Learned regression tree model:\n" + treeModel.toDebugString)
    }
    
    def decision_randomforest(dataset: Dataset[Row], preferredTargets:Seq[String]=Seq(), preferredFeatures:Seq[String]=Seq()) = {
      
        val datasetRDD = Utils.preprocessML(dataset, preferredTargets, preferredFeatures)
        
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.evaluation.RegressionEvaluator
        import org.apache.spark.ml.feature.VectorIndexer
        import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
        
        import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
        import scala.reflect.runtime.{universe => ru}
        
        import org.apache.spark.mllib.linalg.Vectors
        import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
        val observations = datasetRDD.map { x => Vectors.dense(x.features.toArray) }
        val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
        println(summary.mean)  // a dense vector containing the mean value for each column
        println(summary.variance)  // column-wise variance
        println(summary.numNonzeros)  // number of nonzeros in each column
        
        var data = spark.createDataset(datasetRDD)
        
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          //.setMaxCategories(100)
          .fit(data)
        
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        
        val rf = new RandomForestRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
        
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, rf))
        
        val model = pipeline.fit(trainingData)
        val predictions = model.transform(testData)
        predictions.select("prediction", "label", "features").show(5)
        
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        rmse
    }
    
    def randomforest() = {
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.evaluation.RegressionEvaluator
        import org.apache.spark.ml.feature.VectorIndexer
        import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
        
        // Load and parse the data file, converting it to a DataFrame.
        val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(data)
        
        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        
        // Train a RandomForest model.
        val rf = new RandomForestRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
        
        // Chain indexer and forest in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, rf))
        
        // Train model. This also runs the indexer.
        val model = pipeline.fit(trainingData)
        
        // Make predictions.
        val predictions = model.transform(testData)
        
        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)
        
        val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
        println("Learned regression forest model:\n" + rfModel.toDebugString)
    }
    
    def decision_Gradient_boosted_tree(dataset: Dataset[Row], preferredTargets:Seq[String]=Seq(), preferredFeatures:Seq[String]=Seq()) = {
      
        val datasetRDD = Utils.preprocessML(dataset, preferredTargets, preferredFeatures)
        
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.evaluation.RegressionEvaluator
        import org.apache.spark.ml.feature.VectorIndexer
        import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
        
        import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
        import scala.reflect.runtime.{universe => ru}
        
        var data = spark.createDataset(datasetRDD)
        
        import org.apache.spark.ml.feature.Normalizer
        val normalizer = new Normalizer()
          .setInputCol("features")
          .setOutputCol("normFeatures")
          .setP(1.0)
        val l1NormData = normalizer.transform(data)
        val normDF = l1NormData.drop("features").withColumnRenamed("normFeatures", "features")
        
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(100)
          .fit(normDF)
        
        val Array(trainingData, testData) = normDF.randomSplit(Array(0.7, 0.3))
        
        val gbt = new GBTRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
          .setMaxIter(10)
        
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, gbt))
        
        val model = pipeline.fit(trainingData)
        val predictions = model.transform(testData)
        
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        rmse
    }
    
    def Gradient_boosted_tree() = {
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.evaluation.RegressionEvaluator
        import org.apache.spark.ml.feature.VectorIndexer
        import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
        
        // Load and parse the data file, converting it to a DataFrame.
        val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(data)
        
        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        
        // Train a GBT model.
        val gbt = new GBTRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
          .setMaxIter(10)
        
        // Chain indexer and GBT in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, gbt))
        
        // Train model. This also runs the indexer.
        val model = pipeline.fit(trainingData)
        
        // Make predictions.
        val predictions = model.transform(testData)
        
        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)
        
        val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
        println("Learned regression GBT model:\n" + gbtModel.toDebugString)
    }
    
    def Survival_regression() = {
        import org.apache.spark.ml.linalg.Vectors
        import org.apache.spark.ml.regression.AFTSurvivalRegression
        
        val training = spark.createDataFrame(Seq(
          (1.218, 1.0, Vectors.dense(1.560, -0.605)),
          (2.949, 0.0, Vectors.dense(0.346, 2.158)),
          (3.627, 0.0, Vectors.dense(1.380, 0.231)),
          (0.273, 1.0, Vectors.dense(0.520, 1.151)),
          (4.199, 0.0, Vectors.dense(0.795, -0.226))
        )).toDF("label", "censor", "features")
        val quantileProbabilities = Array(0.3, 0.6)
        val aft = new AFTSurvivalRegression()
          .setQuantileProbabilities(quantileProbabilities)
          .setQuantilesCol("quantiles")
        
        val model = aft.fit(training)
        
        // Print the coefficients, intercept and scale parameter for AFT survival regression
        println(s"Coefficients: ${model.coefficients} Intercept: " +
          s"${model.intercept} Scale: ${model.scale}")
        model.transform(training).show(false)
    }
    
    def Isotonic_regression() = {
        import org.apache.spark.ml.regression.IsotonicRegression

        // Loads data.
        val dataset = spark.read.format("libsvm")
          .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")
        
        // Trains an isotonic regression model.
        val ir = new IsotonicRegression()
        val model = ir.fit(dataset)
        
        println(s"Boundaries in increasing order: ${model.boundaries}")
        println(s"Predictions associated with the boundaries: ${model.predictions}")
        
        // Makes predictions.
        model.transform(dataset).show()
    }
  
}