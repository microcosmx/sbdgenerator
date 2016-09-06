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


case class MLFeatureTransform(
    spark: SparkSession)
{
  
    import spark.implicits._
    import org.apache.spark.sql.catalyst.encoders.{OuterScopes, RowEncoder}
    
    def Tokenizer() = {
        import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

        val sentenceDataFrame = spark.createDataFrame(Seq(
          (0, "Hi I heard about Spark"),
          (1, "I wish Java could use case classes"),
          (2, "Logistic,regression,models,are,neat")
        )).toDF("label", "sentence")
        
        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        val regexTokenizer = new RegexTokenizer()
          .setInputCol("sentence")
          .setOutputCol("words")
          .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)
        
        val tokenized = tokenizer.transform(sentenceDataFrame)
        tokenized.select("words", "label").take(3).foreach(println)
        val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
        regexTokenized.select("words", "label").take(3).foreach(println)
    }
    
    def StopWordsRemover() = {
        import org.apache.spark.ml.feature.StopWordsRemover

        val remover = new StopWordsRemover()
          .setInputCol("raw")
          .setOutputCol("filtered")
        
        val dataSet = spark.createDataFrame(Seq(
          (0, Seq("I", "saw", "the", "red", "baloon")),
          (1, Seq("Mary", "had", "a", "little", "lamb"))
        )).toDF("id", "raw")
        
        remover.transform(dataSet).show()
    }
    
    def n_gram() = {
        import org.apache.spark.ml.feature.NGram

        val wordDataFrame = spark.createDataFrame(Seq(
          (0, Array("Hi", "I", "heard", "about", "Spark")),
          (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
          (2, Array("Logistic", "regression", "models", "are", "neat"))
        )).toDF("label", "words")
        
        val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")
        val ngramDataFrame = ngram.transform(wordDataFrame)
        ngramDataFrame.take(3).map(_.getAs[Stream[String]]("ngrams").toList).foreach(println)
    }
    
    def Binarizer() = {
        import org.apache.spark.ml.feature.Binarizer

        val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
        val dataFrame = spark.createDataFrame(data).toDF("label", "feature")
        
        val binarizer: Binarizer = new Binarizer()
          .setInputCol("feature")
          .setOutputCol("binarized_feature")
          .setThreshold(0.5)
        
        val binarizedDataFrame = binarizer.transform(dataFrame)
        val binarizedFeatures = binarizedDataFrame.select("binarized_feature")
        binarizedFeatures.collect().foreach(println)

    }
    
    def PCA() = {
        import org.apache.spark.ml.feature.PCA
        import org.apache.spark.ml.linalg.Vectors
        
        val data = Array(
          Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
          Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
          Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
        )
        val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
        val pca = new PCA()
          .setInputCol("features")
          .setOutputCol("pcaFeatures")
          .setK(3)
          .fit(df)
        val pcaDF = pca.transform(df)
        val result = pcaDF.select("pcaFeatures")
        println(result.head().get(0))
        result.show()
    }
    
    def PolynomialExpansion() = {
        import org.apache.spark.ml.feature.PolynomialExpansion
        import org.apache.spark.ml.linalg.Vectors
        
        val data = Array(
          Vectors.dense(-2.0, 2.3),
          Vectors.dense(0.0, 0.0),
          Vectors.dense(0.6, -1.1)
        )
        val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
        val polynomialExpansion = new PolynomialExpansion()
          .setInputCol("features")
          .setOutputCol("polyFeatures")
          .setDegree(3)
        val polyDF = polynomialExpansion.transform(df)
        polyDF.select("polyFeatures").take(3).foreach(println)
    }
    
    def DCT() = {
        import org.apache.spark.ml.feature.DCT
        import org.apache.spark.ml.linalg.Vectors
        
        val data = Seq(
          Vectors.dense(0.0, 1.0, -2.0, 3.0),
          Vectors.dense(-1.0, 2.0, 4.0, -7.0),
          Vectors.dense(14.0, -2.0, -5.0, 1.0))
        
        val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
        
        val dct = new DCT()
          .setInputCol("features")
          .setOutputCol("featuresDCT")
          .setInverse(false)
        
        val dctDf = dct.transform(df)
        dctDf.select("featuresDCT").show(3)
    }
    
    def StringIndexer() = {
        import org.apache.spark.ml.feature.StringIndexer

        val df = spark.createDataFrame(
          Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
        ).toDF("id", "category")
        
        val indexer = new StringIndexer()
          .setInputCol("category")
          .setOutputCol("categoryIndex")
        
        val indexed = indexer.fit(df).transform(df)
        indexed.show()
        
        indexed.drop("category").withColumnRenamed("categoryIndex", "category").show
    }
    
    def IndexToString() = {
        import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

        val df = spark.createDataFrame(Seq(
          (0, "a"),
          (1, "b"),
          (2, "c"),
          (3, "a"),
          (4, "a"),
          (5, "c")
        )).toDF("id", "category")
        
        val indexer = new StringIndexer()
          .setInputCol("category")
          .setOutputCol("categoryIndex")
          .fit(df)
        val indexed = indexer.transform(df)
        
        val converter = new IndexToString()
          .setInputCol("categoryIndex")
          .setOutputCol("originalCategory")
        
        val converted = converter.transform(indexed)
        converted.select("id", "originalCategory").show()
    }
    
    def OneHotEncoder() = {
        import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

        val df = spark.createDataFrame(Seq(
          (0, "a"),
          (1, "b"),
          (2, "c"),
          (3, "a"),
          (4, "a"),
          (5, "c")
        )).toDF("id", "category")
        
        val indexer = new StringIndexer()
          .setInputCol("category")
          .setOutputCol("categoryIndex")
          .fit(df)
        val indexed = indexer.transform(df)
        
        val encoder = new OneHotEncoder()
          .setInputCol("categoryIndex")
          .setOutputCol("categoryVec")
        val encoded = encoder.transform(indexed)
        encoded.select("id", "categoryVec").show()

    }
    
    def VectorIndexer() = {
        import org.apache.spark.ml.feature.VectorIndexer

        val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        val indexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexed")
          .setMaxCategories(10)
        
        val indexerModel = indexer.fit(data)
        
        val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
        println(s"Chose ${categoricalFeatures.size} categorical features: " +
          categoricalFeatures.mkString(", "))
        
        // Create new column "indexed" with categorical values transformed to indices
        val indexedData = indexerModel.transform(data)
        indexedData.show()
        println(indexedData.head().getAs[String]("features"))
        println("------------------------------------")
        println(indexedData.head().getAs[String]("indexed"))
    }
    
    def Normalizer() = {
        import org.apache.spark.ml.feature.Normalizer

        val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        // Normalize each Vector using $L^1$ norm.
        val normalizer = new Normalizer()
          .setInputCol("features")
          .setOutputCol("normFeatures")
          .setP(1.0)
        
        val l1NormData = normalizer.transform(dataFrame)
        l1NormData.show()
        println(l1NormData.head().getAs[String]("features"))
        println("------------------------------------")
        println(l1NormData.head().getAs[String]("normFeatures"))
        
        // Normalize each Vector using $L^\infty$ norm.
        val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
        lInfNormData.show()
    }
    
    def StandardScaler() = {
        import org.apache.spark.ml.feature.StandardScaler

        val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        val scaler = new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithStd(true)
          .setWithMean(false)
        
        // Compute summary statistics by fitting the StandardScaler.
        val scalerModel = scaler.fit(dataFrame)
        
        // Normalize each feature to have unit standard deviation.
        val scaledData = scalerModel.transform(dataFrame)
        scaledData.show()
        
        println(scaledData.head().getAs[String]("features"))
        println("------------------------------------")
        println(scaledData.head().getAs[String]("scaledFeatures"))
    }
    
    def MinMaxScaler() = {
        import org.apache.spark.ml.feature.MinMaxScaler

        val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        
        val scaler = new MinMaxScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
        
        // Compute summary statistics and generate MinMaxScalerModel
        val scalerModel = scaler.fit(dataFrame)
        
        // rescale each feature to range [min, max].
        val scaledData = scalerModel.transform(dataFrame)
        scaledData.show()
        
        println(scaledData.head().getAs[String]("features"))
        println("------------------------------------")
        println(scaledData.head().getAs[String]("scaledFeatures"))
    }
    
    def MaxAbsScaler() = {
        import org.apache.spark.ml.feature.MaxAbsScaler

        val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
        val scaler = new MaxAbsScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
        
        // Compute summary statistics and generate MaxAbsScalerModel
        val scalerModel = scaler.fit(dataFrame)
        
        // rescale each feature to range [-1, 1]
        val scaledData = scalerModel.transform(dataFrame)
        scaledData.show()
        
        println(scaledData.head().getAs[String]("features"))
        println("------------------------------------")
        println(scaledData.head().getAs[String]("scaledFeatures"))
    }
    
    def Bucketizer() = {
        import org.apache.spark.ml.feature.Bucketizer

        val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
        
        val data = Array(-0.5, -0.3, 0.0, 0.2)
        val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
        
        val bucketizer = new Bucketizer()
          .setInputCol("features")
          .setOutputCol("bucketedFeatures")
          .setSplits(splits)
        
        // Transform original data into its bucket index.
        val bucketedData = bucketizer.transform(dataFrame)
        bucketedData.show()
    }
    
    def ElementwiseProduct() = {
        import org.apache.spark.ml.feature.ElementwiseProduct
        import org.apache.spark.ml.linalg.Vectors
        
        // Create some vector data; also works for sparse vectors
        val dataFrame = spark.createDataFrame(Seq(
          ("a", Vectors.dense(1.0, 2.0, 3.0)),
          ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")
        
        val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
        val transformer = new ElementwiseProduct()
          .setScalingVec(transformingVector)
          .setInputCol("vector")
          .setOutputCol("transformedVector")
        
        // Batch transform the vectors to create new column:
        transformer.transform(dataFrame).show()
    }
    
    def SQLTransformer() = {
        import org.apache.spark.ml.feature.SQLTransformer

        val df = spark.createDataFrame(
          Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")
        
        val sqlTrans = new SQLTransformer().setStatement(
          "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
        
        sqlTrans.transform(df).show()
    }
    
    def VectorAssembler() = {
        import org.apache.spark.ml.feature.VectorAssembler
        import org.apache.spark.ml.linalg.Vectors
        
        val dataset = spark.createDataFrame(
          Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
        ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
        
        val assembler = new VectorAssembler()
          .setInputCols(Array("hour", "mobile", "userFeatures"))
          .setOutputCol("features")
        
        val output = assembler.transform(dataset)
        println(output.select("features", "clicked").first())
    }
    
    def QuantileDiscretizer() = {
        import org.apache.spark.ml.feature.QuantileDiscretizer

        val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
        var df = spark.createDataFrame(data).toDF("id", "hour")
        
        val discretizer = new QuantileDiscretizer()
          .setInputCol("hour")
          .setOutputCol("result")
          .setNumBuckets(3)
        
        val result = discretizer.fit(df).transform(df)
        result.show()
    }
  
}