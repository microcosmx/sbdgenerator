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


case class StatGenetor(
    spark: SparkSession)
{
    val sc = spark.sparkContext
  
    import spark.implicits._
    import spark.sqlContext.implicits._
    import org.apache.spark.sql.catalyst.encoders._
    
    def statCols() = {
      
        import org.apache.spark.mllib.linalg.Vectors
        import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
        
        val observations = sc.parallelize(
          Seq(
            Vectors.dense(1.0, 10.0, 100.0),
            Vectors.dense(2.0, 20.0, 200.0),
            Vectors.dense(3.0, 30.0, 300.0)
          )
        )
        
        // Compute column summary statistics.
        val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
        println(summary.mean)  // a dense vector containing the mean value for each column
        println(summary.variance)  // column-wise variance
        println(summary.numNonzeros)  // number of nonzeros in each column
    }
    
    def statFeaturess(dataset:Dataset[Row] ) = {
      
        import org.apache.spark.mllib.linalg.Vectors
        import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
        
        import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
        import scala.reflect.runtime.{universe => ru}
        
        var fs = dataset.schema.fields
        var fsNames = fs.map(_.name).toSeq
        
        val datasetRDD = dataset.rdd.map { row => 
              val result = row.toSeq.map(x=>{
                if(x.isInstanceOf[String]) 0.0 
                else if(x.isInstanceOf[Int]) x.asInstanceOf[Int].toDouble
                else if(x.isInstanceOf[Long]) x.asInstanceOf[Long].toDouble
                else if(x.isInstanceOf[Double]) x.asInstanceOf[Double].toDouble
                else 0.0
              })
              
              val features = Vectors.dense(result.toArray)
              features
         }
        
        // Compute column summary statistics.
        val summary: MultivariateStatisticalSummary = Statistics.colStats(datasetRDD)
        println(summary.mean)  // a dense vector containing the mean value for each column
        println(summary.variance)  // column-wise variance
        println(summary.numNonzeros)  // number of nonzeros in each column
        
        summary
    }
    
    def Correlations() = {
        import org.apache.spark.mllib.linalg._
        import org.apache.spark.mllib.stat.Statistics
        import org.apache.spark.rdd.RDD
        
        val seriesX: RDD[Double] = sc.parallelize(Array(1, 2, 3, 3, 5))  // a series
        // must have the same number of partitions and cardinality as seriesX
        val seriesY: RDD[Double] = sc.parallelize(Array(11, 22, 33, 33, 555))
        
        // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method. If a
        // method is not specified, Pearson's method will be used by default.
        val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")
        println(s"Correlation is: $correlation")
        
        val data: RDD[Vector] = sc.parallelize(
          Seq(
            Vectors.dense(1.0, 10.0, 100.0),
            Vectors.dense(2.0, 20.0, 200.0),
            Vectors.dense(5.0, 33.0, 366.0))
        )  // note that each Vector is a row and not a column
        
        // calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method
        // If a method is not specified, Pearson's method will be used by default.
        val correlMatrix: Matrix = Statistics.corr(data, "pearson")
        println(correlMatrix.toString)
    }
    
    def sampling() = {
        // an RDD[(K, V)] of any key value pairs
        val data = sc.parallelize(
          Seq((1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (2, 'e'), (3, 'f')))
        
        // specify the exact fraction desired from each key
        val fractions = Map(1 -> 0.1, 2 -> 0.6, 3 -> 0.3)
        
        // Get an approximate sample from each stratum
        val approxSample = data.sampleByKey(withReplacement = false, fractions = fractions)
        println(approxSample.collect.toSeq)
        // Get an exact sample from each stratum
        val exactSample = data.sampleByKeyExact(withReplacement = false, fractions = fractions)
        println(exactSample.collect.toSeq)
    }
    
    def Hypothesis_testing() = {
        import org.apache.spark.mllib.linalg._
        import org.apache.spark.mllib.regression.LabeledPoint
        import org.apache.spark.mllib.stat.Statistics
        import org.apache.spark.mllib.stat.test.ChiSqTestResult
        import org.apache.spark.rdd.RDD
        
        // a vector composed of the frequencies of events
        val vec: Vector = Vectors.dense(0.1, 0.15, 0.2, 0.3, 0.25)
        
        // compute the goodness of fit. If a second vector to test against is not supplied
        // as a parameter, the test runs against a uniform distribution.
        val goodnessOfFitTestResult = Statistics.chiSqTest(vec)
        // summary of the test including the p-value, degrees of freedom, test statistic, the method
        // used, and the null hypothesis.
        println(s"$goodnessOfFitTestResult\n")
        
        // a contingency matrix. Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        val mat: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
        
        // conduct Pearson's independence test on the input contingency matrix
        val independenceTestResult = Statistics.chiSqTest(mat)
        // summary of the test including the p-value, degrees of freedom
        println(s"$independenceTestResult\n")
        
        val obs: RDD[LabeledPoint] =
          sc.parallelize(
            Seq(
              LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
              LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 0.0)),
              LabeledPoint(-1.0, Vectors.dense(-1.0, 0.0, -0.5)
              )
            )
          ) // (feature, label) pairs.
        
        // The contingency table is constructed from the raw (feature, label) pairs and used to conduct
        // the independence test. Returns an array containing the ChiSquaredTestResult for every feature
        // against the label.
        val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)
        featureTestResults.zipWithIndex.foreach { case (k, v) =>
          println("Column " + (v + 1).toString + ":")
          println(k)
        }  // summary of the test
    }
    
    def Hypothesis_testing2() = {
        import org.apache.spark.mllib.stat.Statistics
        import org.apache.spark.rdd.RDD
        
        val data: RDD[Double] = sc.parallelize(Seq(0.1, 0.15, 0.2, 0.3, 0.25))  // an RDD of sample data
        
        // run a KS test for the sample versus a standard normal distribution
        val testResult = Statistics.kolmogorovSmirnovTest(data, "norm", 0, 1)
        // summary of the test including the p-value, test statistic, and null hypothesis if our p-value
        // indicates significance, we can reject the null hypothesis.
        println(testResult)
        println()
        
        // perform a KS test using a cumulative distribution function of our making
        val myCDF = Map(0.1 -> 0.2, 0.15 -> 0.6, 0.2 -> 0.05, 0.3 -> 0.05, 0.25 -> 0.1)
        val testResult2 = Statistics.kolmogorovSmirnovTest(data, myCDF)
        println(testResult2)
    }
    
//    def Streaming_Significance_Testing() = {
//        val data = ssc.textFileStream(dataDir).map(line => line.split(",") match {
//          case Array(label, value) => BinarySample(label.toBoolean, value.toDouble)
//        })
//        
//        val streamingTest = new StreamingTest()
//          .setPeacePeriod(0)
//          .setWindowSize(0)
//          .setTestMethod("welch")
//        
//        val out = streamingTest.registerStream(data)
//        out.print()
//      
//    }
    
    def randomData() = {
        import org.apache.spark.SparkContext
        import org.apache.spark.mllib.random.RandomRDDs._
        
        // Generate a random double RDD that contains 1 million i.i.d. values drawn from the
        // standard normal distribution `N(0, 1)`, evenly distributed in 10 partitions.
        val u = normalRDD(sc, 1000000L, 10)
        // Apply a transform to get a random double RDD following `N(1, 4)`.
        val v = u.map(x => 1.0 + 2.0 * x)
        
        println(v.collect.toSeq)
    }
    
    def Kernel_density_estimation() = {
        import org.apache.spark.mllib.stat.KernelDensity
        import org.apache.spark.rdd.RDD
        
        // an RDD of sample data
        val data: RDD[Double] = sc.parallelize(Seq(1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9))
        
        // Construct the density estimator with the sample data and a standard deviation
        // for the Gaussian kernels
        val kd = new KernelDensity()
          .setSample(data)
          .setBandwidth(3.0)
        
        // Find density estimates for the given values
        val densities = kd.estimate(Array(-1.0, 2.0, 5.0))
        
        println(densities.toSeq)
    }
    
  
}