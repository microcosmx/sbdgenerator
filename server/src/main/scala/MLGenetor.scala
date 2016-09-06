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


case class MLGenetor(
    spark: SparkSession)
{
  
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
    
    def mlpipline() = {
      
        // Prepare training data from a list of (label, features) tuples.
        val training = spark.createDataFrame(Seq(
          (1.0, Vectors.dense(0.0, 1.1, 0.1)),
          (0.0, Vectors.dense(2.0, 1.0, -1.0)),
          (0.0, Vectors.dense(2.0, 1.3, 1.0)),
          (1.0, Vectors.dense(0.0, 1.2, -0.5))
        )).toDF("label", "features")
        
        // Create a LogisticRegression instance. This instance is an Estimator.
        val lr = new LogisticRegression()
        // Print out the parameters, documentation, and any default values.
        println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
        
        // We may set parameters using setter methods.
        lr.setMaxIter(10)
          .setRegParam(0.01)
        
        // Learn a LogisticRegression model. This uses the parameters stored in lr.
        val model1 = lr.fit(training)
        // Since model1 is a Model (i.e., a Transformer produced by an Estimator),
        // we can view the parameters it used during fit().
        // This prints the parameter (name: value) pairs, where names are unique IDs for this
        // LogisticRegression instance.
        println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)
        
        // We may alternatively specify parameters using a ParamMap,
        // which supports several methods for specifying parameters.
        val paramMap = ParamMap(lr.maxIter -> 20)
          .put(lr.maxIter, 30)  // Specify 1 Param. This overwrites the original maxIter.
          .put(lr.regParam -> 0.1, lr.threshold -> 0.55)  // Specify multiple Params.
        
        // One can also combine ParamMaps.
        val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")  // Change output column name.
        val paramMapCombined = paramMap ++ paramMap2
        
        // Now learn a new model using the paramMapCombined parameters.
        // paramMapCombined overrides all parameters set earlier via lr.set* methods.
        val model2 = lr.fit(training, paramMapCombined)
        println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)
        
        // Prepare test data.
        val test = spark.createDataFrame(Seq(
          (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
          (0.0, Vectors.dense(3.0, 2.0, -0.1)),
          (1.0, Vectors.dense(0.0, 2.2, -1.5))
        )).toDF("label", "features")
        
        // Make predictions on test data using the Transformer.transform() method.
        // LogisticRegression.transform will only use the 'features' column.
        // Note that model2.transform() outputs a 'myProbability' column instead of the usual
        // 'probability' column since we renamed the lr.probabilityCol parameter previously.
        model2.transform(test)
          .select("features", "label", "myProbability", "prediction")
          .collect()
          .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
            println(s"($features, $label) -> prob=$prob, prediction=$prediction")
          }
    }
    
    
    def mlPipline2() = {
        val training = spark.createDataFrame(Seq(
          (0L, "a b c d e spark", 1.0),
          (1L, "b d", 0.0),
          (2L, "spark f g h", 1.0),
          (3L, "hadoop mapreduce", 0.0)
        )).toDF("id", "text", "label")
        
        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        val tokenizer = new Tokenizer()
          .setInputCol("text")
          .setOutputCol("words")
        val hashingTF = new HashingTF()
          .setNumFeatures(1000)
          .setInputCol(tokenizer.getOutputCol)
          .setOutputCol("features")
        val lr = new LogisticRegression()
          .setMaxIter(100)
          .setRegParam(0.01)
        val pipeline = new Pipeline()
          .setStages(Array(tokenizer, hashingTF, lr))
        
        // Fit the pipeline to training documents.
        val model = pipeline.fit(training)
        
        // Now we can optionally save the fitted pipeline to disk
        //model.write.overwrite().save("/tmp/spark-logistic-regression-model")
        
        // We can also save this unfit pipeline to disk
        //pipeline.write.overwrite().save("/tmp/unfit-lr-model")
        
        // And load it back in during production
        //val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")
        
        // Prepare test documents, which are unlabeled (id, text) tuples.
        val test = spark.createDataFrame(Seq(
          (4L, "spark i j k"),
          (5L, "l m n"),
          (6L, "mapreduce spark"),
          (7L, "apache hadoop")
        )).toDF("id", "text")
        
        // Make predictions on test documents.
        model.transform(test)
          .select("id", "text", "probability", "prediction")
          .collect()
          .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
            println(s"($id, $text) --> prob=$prob, prediction=$prediction")
          }
    }
    
    
    //choose the best parameters(from the provided existing params) for specific model or pipline
    def modelselection() = {
        
        // Prepare training data from a list of (id, text, label) tuples.
        val training = spark.createDataFrame(Seq(
          (0L, "a b c d e spark", 1.0),
          (1L, "b d", 0.0),
          (2L, "spark f g h", 1.0),
          (3L, "hadoop mapreduce", 0.0),
          (4L, "b spark who", 1.0),
          (5L, "g d a y", 0.0),
          (6L, "spark fly", 1.0),
          (7L, "was mapreduce", 0.0),
          (8L, "e spark program", 1.0),
          (9L, "a e c l", 0.0),
          (10L, "spark compile", 1.0),
          (11L, "hadoop software", 0.0)
        )).toDF("id", "text", "label")
        
        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        val tokenizer = new Tokenizer()
          .setInputCol("text")
          .setOutputCol("words")
        val hashingTF = new HashingTF()
          .setInputCol(tokenizer.getOutputCol)
          .setOutputCol("features")
        val lr = new LogisticRegression()
          .setMaxIter(10)
        val pipeline = new Pipeline()
          .setStages(Array(tokenizer, hashingTF, lr))
        
        // We use a ParamGridBuilder to construct a grid of parameters to search over.
        // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
        // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
        val paramGrid = new ParamGridBuilder()
          .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
          .addGrid(lr.regParam, Array(0.1, 0.01))
          .build()
        
        // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
        // This will allow us to jointly choose parameters for all Pipeline stages.
        // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
        // is areaUnderROC.
        val cv = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(new BinaryClassificationEvaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(2)  // Use 3+ in practice
        
        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(training)
        
        // Prepare test documents, which are unlabeled (id, text) tuples.
        val test = spark.createDataFrame(Seq(
          (4L, "spark i j k"),
          (5L, "l m n"),
          (6L, "mapreduce spark"),
          (7L, "apache hadoop")
        )).toDF("id", "text")
        
        // Make predictions on test documents. cvModel uses the best model found (lrModel).
        cvModel.transform(test)
          .select("id", "text", "probability", "prediction")
          .collect()
          .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
            println(s"($id, $text) --> prob=$prob, prediction=$prediction")
          }
    }
    
    def decisionTreeMl(data: Dataset[Row]) = {
        
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
    
    
    def decisionTreeMSE(dataset: Dataset[Row]) = {
        import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
        import scala.reflect.runtime.{universe => ru}
        
        var fs = dataset.schema.fields
        var dataset1 = dataset
        fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="string").map(x=>{
             val indexer = new StringIndexer()
              .setInputCol(x._1.name)
              .setOutputCol(x._1.name+"_index")
            val indexed = indexer.fit(dataset1).transform(dataset1)
            dataset1 = indexed.drop(x._1.name).withColumnRenamed(x._1.name+"_index", x._1.name)
        })
        fs = dataset1.schema.fields
        
        val datasetRDD = dataset1.rdd.map { row => 
              val typeMirror = ru.runtimeMirror(row.getClass.getClassLoader)
              val instanceMirror = typeMirror.reflect(row)
              val fsValue = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="double").map(x => {
                  val methodX = ru.typeOf[Row].declaration(ru.newTermName("get")).asMethod
                  val thevalue = instanceMirror.reflectMethod(methodX)(x._2)
                  if(thevalue==null) 0.0 else thevalue.asInstanceOf[Double]
              }).toSeq
              val fsValueInt = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="int").map(x => {
                  val methodX = ru.typeOf[Row].declaration(ru.newTermName("get")).asMethod
                  val thevalue = instanceMirror.reflectMethod(methodX)(x._2)
                  if(thevalue==null) 0.0 else thevalue.asInstanceOf[Int].toDouble
              }).toSeq
              val fsValueLong = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="long").map(x => {
                  val methodX = ru.typeOf[Row].declaration(ru.newTermName("get")).asMethod
                  val thevalue = instanceMirror.reflectMethod(methodX)(x._2)
                  if(thevalue==null) 0.0 else thevalue.asInstanceOf[Long].toDouble
              }).toSeq
              
              val vectors = fsValue ++ fsValueInt ++ fsValueLong
              val label = vectors.head
              val fvector = vectors.tail
              val features = Vectors.dense(fvector.head, fvector.tail:_*)
              LabeledPoint(
                label, features
              )
         }
        
        var data = spark.createDataset(datasetRDD)//(Encoders.kryo[LabeledPoint])
        
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          //.setMaxCategories(4)
          .fit(data)
        
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        
        val dt = new DecisionTreeRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")
        
        // Chain indexer and tree in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, dt))
        
        val model = pipeline.fit(trainingData)
        val predictions = model.transform(testData)
        //predictions.select("prediction", "label", "features").show(5)
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        
        var rmse = 0.0
        try {
          rmse = evaluator.evaluate(predictions)
        }
        catch {
            case t: Throwable => println(s"-----${t.printStackTrace()}--------")
        }
        finally{
            rmse = 0.0
        }
        
        //val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
        //println("Learned regression tree model:\n" + treeModel.toDebugString)
        
        rmse
    }
    
    
    def decisionPipline(dataset: Dataset[Row]) = {
        
        import scala.reflect.runtime.{universe => ru}
        val fs = dataset.schema.fields
        val datasetDF = dataset.rdd.map { row => 
              val fsValue = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="int").map(x => {
                  val thevalue = row.get(x._2)
                  if(thevalue==null) 0 else thevalue.asInstanceOf[Int]
              }).toSeq
              (fsValue.tail.mkString(" "), fsValue.head%2)
         }.toDF("text", "label")
         
         //datasetDF.show
        
        var data = datasetDF
        
        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        val tokenizer = new Tokenizer()
          .setInputCol("text")
          .setOutputCol("words")
        val hashingTF = new HashingTF()
          .setNumFeatures(1000)
          .setInputCol(tokenizer.getOutputCol)
          .setOutputCol("features")
        val lr = new LogisticRegression()
          .setMaxIter(100)
          .setRegParam(0.01)
        val pipeline = new Pipeline()
          .setStages(Array(tokenizer, hashingTF, lr))
        
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        val model = pipeline.fit(trainingData)
        val predictions = model.transform(testData)
        
        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        println("RMSE: " + rmse)
        rmse
    }
    
    
    
    //extra algorithm
    def decisiontree() = {
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.classification.DecisionTreeClassificationModel
        import org.apache.spark.ml.classification.DecisionTreeClassifier
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
        
        // Load the data stored in LIBSVM format as a DataFrame.
        val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        data.printSchema
        data.show
        
        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(data)
        // Automatically identify categorical features, and index them.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
          .fit(data)
        
        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        
        // Train a DecisionTree model.
        val dt = new DecisionTreeClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")
        
        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labels)
        
        // Chain indexers and tree in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
        
        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainingData)
        
        // Make predictions.
        val predictions = model.transform(testData)
        
        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5)
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println("Test Error = " + (1.0 - accuracy))
        
        val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
        println("Learned classification tree model:\n" + treeModel.toDebugString)
      
    }
    
    
    def randomforest() = {
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
        
        // Load and parse the data file, converting it to a DataFrame.
        //val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
        
        data.printSchema
        data.show
        
        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(data)
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
        val rf = new RandomForestClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")
          .setNumTrees(10)
        
        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labels)
        
        // Chain indexers and forest in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
        
        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainingData)
        
        // Make predictions.
        val predictions = model.transform(testData)
        
        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5)
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println("Test Error = " + (1.0 - accuracy))
        
        val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
        println("Learned classification forest model:\n" + rfModel.toDebugString)
    }
    
    def Gradientboostedtree() = {
        import org.apache.spark.ml.Pipeline
        import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
        
        // Load and parse the data file, converting it to a DataFrame.
        val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(data)
        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(data)
          
        data.printSchema
        data.show
        
        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
        
        // Train a GBT model.
        val gbt = new GBTClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")
          .setMaxIter(10)
        
        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labels)
        
        // Chain indexers and GBT in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
        
        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainingData)
        
        // Make predictions.
        val predictions = model.transform(testData)
        
        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5)
        
        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println("Test Error = " + (1.0 - accuracy))
        
        val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
        println("Learned classification GBT model:\n" + gbtModel.toDebugString)
    }
    
    
    def decision_Multilayer_perceptron_classifier(dataset: Dataset[Row]) = {
        import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
        import scala.reflect.runtime.{universe => ru}
        
        var fs = dataset.schema.fields
        var dataset1 = dataset
        fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="string").map(x=>{
             val indexer = new StringIndexer()
              .setInputCol(x._1.name)
              .setOutputCol(x._1.name+"_index")
            val indexed = indexer.fit(dataset1).transform(dataset1)
            dataset1 = indexed.drop(x._1.name).withColumnRenamed(x._1.name+"_index", x._1.name)
        })
        fs = dataset1.schema.fields
        
        val datasetRDD = dataset1.rdd.map { row => 
              val typeMirror = ru.runtimeMirror(row.getClass.getClassLoader)
              val instanceMirror = typeMirror.reflect(row)
              val fsValue = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="double").map(x => {
                  val thevalue = row.get(x._2)
                  if(thevalue==null) 0.0 else thevalue.asInstanceOf[Double]
              }).toSeq
              val fsValueInt = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="int").map(x => {
                  val thevalue = row.get(x._2)
                  if(thevalue==null) 0.0 else thevalue.asInstanceOf[Int].toDouble
              }).toSeq
              val fsValueLong = fs.zipWithIndex.filter(x=>x._1.dataType.simpleString=="long").map(x => {
                  val thevalue = row.get(x._2)
                  if(thevalue==null) 0.0 else thevalue.asInstanceOf[Long].toDouble
              }).toSeq
              
              val vectors = fsValue ++ fsValueInt ++ fsValueLong
              val label = vectors.head
              val fvector = vectors.tail
              val features = Vectors.dense(fvector.head, fvector.tail:_*)
              LabeledPoint(
                label, features
              )
         }
        
        var data = spark.createDataset(datasetRDD)
        
        val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
        val train = splits(0)
        val test = splits(1)
        // specify layers for the neural network:
        // input layer of size 4 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)
        val layers = Array[Int](4, 5, 4, 3)
        // create the trainer and set its parameters
        val trainer = new MultilayerPerceptronClassifier()
          .setLayers(layers)
          .setBlockSize(128)
          .setSeed(1234L)
          .setMaxIter(100)
        // train the model
        val model = trainer.fit(train)
        // compute accuracy on the test set
        val result = model.transform(test)
        val predictionAndLabels = result.select("prediction", "label")
        val evaluator = new MulticlassClassificationEvaluator()
          .setMetricName("accuracy")
        println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
        val accuracy = evaluator.evaluate(predictionAndLabels)
        accuracy
    }
    
    def Multilayer_perceptron_classifier() = {
        import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        
        // Load the data stored in LIBSVM format as a DataFrame.
        val data = spark.read.format("libsvm")
          .load("data/mllib/sample_multiclass_classification_data.txt")
        // Split the data into train and test
        val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
        val train = splits(0)
        val test = splits(1)
        // specify layers for the neural network:
        // input layer of size 4 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)
        val layers = Array[Int](4, 5, 4, 3)
        // create the trainer and set its parameters
        val trainer = new MultilayerPerceptronClassifier()
          .setLayers(layers)
          .setBlockSize(128)
          .setSeed(1234L)
          .setMaxIter(100)
        // train the model
        val model = trainer.fit(train)
        // compute accuracy on the test set
        val result = model.transform(test)
        val predictionAndLabels = result.select("prediction", "label")
        val evaluator = new MulticlassClassificationEvaluator()
          .setMetricName("accuracy")
        println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
    }
    
    def One_vs_Rest_classifier() = {
        import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        
        // load data file.
        val inputData = spark.read.format("libsvm")
          .load("data/mllib/sample_multiclass_classification_data.txt")
        
        // generate the train/test split.
        val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
        
        // instantiate the base classifier
        val classifier = new LogisticRegression()
          .setMaxIter(10)
          .setTol(1E-6)
          .setFitIntercept(true)
        
        // instantiate the One Vs Rest Classifier.
        val ovr = new OneVsRest().setClassifier(classifier)
        
        // train the multiclass model.
        val ovrModel = ovr.fit(train)
        
        // score the model on test data.
        val predictions = ovrModel.transform(test)
        
        // obtain evaluator.
        val evaluator = new MulticlassClassificationEvaluator()
          .setMetricName("accuracy")
        
        // compute the classification error on test data.
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test Error : ${1 - accuracy}")
    }
    
    def Naive_Bayes() = {
        import org.apache.spark.ml.classification.NaiveBayes
        import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
        
        // Load the data stored in LIBSVM format as a DataFrame.
        val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        // Split the data into training and test sets (30% held out for testing)
        val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
        
        // Train a NaiveBayes model.
        val model = new NaiveBayes()
          .fit(trainingData)
        
        // Select example rows to display.
        val predictions = model.transform(testData)
        predictions.show()
        
        // Select (prediction, true label) and compute test error
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println("Accuracy: " + accuracy)
    }
  
}