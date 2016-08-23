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


case class MLLibGenerator(
    spark: SparkSession)
{
    val sc = spark.sparkContext
  
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
    
    def SVMs() = {
        import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
        import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
        import org.apache.spark.mllib.util.MLUtils
        
        // Load training data in LIBSVM format.
        val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
        
        // Split data into training (60%) and test (40%).
        val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        // Run training algorithm to build the model
        val numIterations = 100
        
        import org.apache.spark.mllib.optimization.L1Updater

        val svmAlg = new SVMWithSGD()
        svmAlg.optimizer
          .setNumIterations(200)
          .setRegParam(0.1)
          .setUpdater(new L1Updater)
        val modelL1 = svmAlg.run(training)

        val model = SVMWithSGD.train(training, numIterations)
        
        // Clear the default threshold.
        model.clearThreshold()
        
        // Compute raw scores on the test set.
        val scoreAndLabels = test.map { point =>
          val score = model.predict(point.features)
          (score, point.label)
        }
        
        // Get evaluation metrics.
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auROC = metrics.areaUnderROC()
        
        println("Area under ROC = " + auROC)
        
        metrics.scoreAndLabels.toDS.show
        
        // Save and load model
        //model.save(sc, "target/tmp/scalaSVMWithSGDModel")
        //val sameModel = SVMModel.load(sc, "target/tmp/scalaSVMWithSGDModel")
    }
    
    
//    def Power_iteration_clustering() = {
//        import org.apache.spark.mllib.clustering.PowerIterationClustering
//
//        val circlesRdd = generateCirclesRdd(sc, params.k, params.numPoints)
//        val model = new PowerIterationClustering()
//          .setK(params.k)
//          .setMaxIterations(params.maxIterations)
//          .setInitializationMode("degree")
//          .run(circlesRdd)
//        
//        val clusters = model.assignments.collect().groupBy(_.cluster).mapValues(_.map(_.id))
//        val assignments = clusters.toList.sortBy { case (k, v) => v.length }
//        val assignmentsStr = assignments
//          .map { case (k, v) =>
//            s"$k -> ${v.sorted.mkString("[", ",", "]")}"
//          }.mkString(", ")
//        val sizesStr = assignments.map {
//          _._2.length
//        }.sorted.mkString("(", ",", ")")
//        println(s"Cluster assignments: $assignmentsStr\ncluster sizes: $sizesStr")
//    }
    
    
    //Singular value decomposition
    def SVD_Example() = {
        import org.apache.spark.mllib.linalg.Matrix
        import org.apache.spark.mllib.linalg.SingularValueDecomposition
        import org.apache.spark.mllib.linalg.Vector
        import org.apache.spark.mllib.linalg.Vectors
        import org.apache.spark.mllib.linalg.distributed.RowMatrix
        
        val data = Array(
          Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
          Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
          Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0))
        
        val dataRDD = sc.parallelize(data, 2)
        
        val mat: RowMatrix = new RowMatrix(dataRDD)
        
        // Compute the top 5 singular values and corresponding singular vectors.
        val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(5, computeU = true)
        val U: RowMatrix = svd.U  // The U factor is a RowMatrix.
        val s: Vector = svd.s  // The singular values are stored in a local dense vector.
        val V: Matrix = svd.V  // The V factor is a local dense matrix.
        
        //println(U)
        println(s)
        //println(V)
    }
    
    def Principal_component_analysis() = {
        import org.apache.spark.mllib.linalg.Matrix
        import org.apache.spark.mllib.linalg.Vectors
        import org.apache.spark.mllib.linalg.distributed.RowMatrix
        
        val data = Array(
          Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
          Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
          Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0))
        
        val dataRDD = sc.parallelize(data, 2)
        
        val mat: RowMatrix = new RowMatrix(dataRDD)
        
        // Compute the top 4 principal components.
        // Principal components are stored in a local dense matrix.
        val pc: Matrix = mat.computePrincipalComponents(4)
        println(pc)
        
        // Project the rows to the linear space spanned by the top 4 principal components.
        val projected: RowMatrix = mat.multiply(pc)
        
        println(projected.rows.collect.toSeq)
    }
    
    def Principal_component_analysis_2() = {
        import org.apache.spark.mllib.feature.PCA
        import org.apache.spark.mllib.linalg.Vectors
        import org.apache.spark.mllib.regression.LabeledPoint
        import org.apache.spark.rdd.RDD
        
        val data: RDD[LabeledPoint] = sc.parallelize(Seq(
          new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 1)),
          new LabeledPoint(1, Vectors.dense(1, 1, 0, 1, 0)),
          new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0)),
          new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 0)),
          new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0))))
        
        // Compute the top 5 principal components.
        val pca = new PCA(5).fit(data.map(_.features))
        
        // Project vectors to the linear space spanned by the top 5 principal
        // components, keeping the label
        val projected = data.map(p => p.copy(features = pca.transform(p.features)))
        
        println(projected.collect.toSeq)
    }
    
    
    
    
    
    
    
    //associations
    def FP_growth() = {
        import org.apache.spark.mllib.fpm.FPGrowth
        import org.apache.spark.rdd.RDD
        
        val data = sc.textFile("data/mllib/sample_fpgrowth.txt")
        
        val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' '))
        
        val fpg = new FPGrowth()
          .setMinSupport(0.2)
          .setNumPartitions(10)
        val model = fpg.run(transactions)
        
        model.freqItemsets.collect().foreach { itemset =>
          println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
        }
        
        val minConfidence = 0.8
        model.generateAssociationRules(minConfidence).collect().foreach { rule =>
          println(
            rule.antecedent.mkString("[", ",", "]")
              + " => " + rule.consequent .mkString("[", ",", "]")
              + ", " + rule.confidence)
        }

    }
    
    def Association_Rules() = {
        import org.apache.spark.mllib.fpm.AssociationRules
        import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
        
        val freqItemsets = sc.parallelize(Seq(
          new FreqItemset(Array("a"), 15L),
          new FreqItemset(Array("b"), 35L),
          new FreqItemset(Array("a", "b"), 12L)
        ))
        
        val ar = new AssociationRules()
          .setMinConfidence(0.8)
        val results = ar.run(freqItemsets)
        
        results.collect().foreach { rule =>
          println("[" + rule.antecedent.mkString(",")
            + "=>"
            + rule.consequent.mkString(",") + "]," + rule.confidence)
        }
    }
    
    def PrefixSpan() = {
        import org.apache.spark.mllib.fpm.PrefixSpan

        val sequences = sc.parallelize(Seq(
          Array(Array(1, 2), Array(3)),
          Array(Array(1), Array(3, 2), Array(1, 2)),
          Array(Array(1, 2), Array(5)),
          Array(Array(6))
        ), 2).cache()
        val prefixSpan = new PrefixSpan()
          .setMinSupport(0.5)
          .setMaxPatternLength(5)
        val model = prefixSpan.run(sequences)
        model.freqSequences.collect().foreach { freqSequence =>
          println(
            freqSequence.sequence.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]") +
              ", " + freqSequence.freq)
        }
    }
  
}