import akka.actor._
import akka.io.IO
import akka.pattern._
import akka.routing._
import akka.util.Timeout
import org.apache.hadoop.fs._
import org.apache.avro._
import org.apache.avro.file._
import org.apache.avro.reflect._
import org.apache.hadoop.fs._
import org.apache.hadoop.conf._
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.storage._
import org.apache.spark.streaming._
import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import spray.can.Http
import spray.json._
import DefaultJsonProtocol._
import org.scalatest._
import scala.annotation._
import scala.math.Ordered.orderingToOrdered
import scala.math.Ordering.Implicits._
import scala.util.hashing._
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.mllib.classification.{SVMModel,SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType,StructField,StringType};

import org.apache.spark.SparkException
import org.apache.spark.sql.catalyst.plans.logical.{OneRowRelation, Union}
import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeExec, ReusedExchangeExec, ShuffleExchange}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

object GA {
  
  implicit lazy val system = ActorSystem()
  implicit val timeout: Timeout = 1.minute
  
  import org.apache.log4j.Logger
  import org.apache.log4j.Level
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)
  Logger.getLogger("parquet.hadoop").setLevel(Level.WARN)

  val fs_conf = new org.apache.hadoop.conf.Configuration
  val fs = FileSystem.get(fs_conf)

  val conf = new org.apache.spark.SparkConf
  conf.set("spark.master", "local[*]")
  //conf.set("spark.master", "spark://192.168.20.17:7070")
  conf.set("spark.app.name", "Aliyun")
  conf.set("spark.ui.port", "55555")
  conf.set("spark.default.parallelism", "10")
  conf.set("spark.sql.shuffle.partitions", "10")
  conf.set("spark.sql.shuffle.partitions", "1")
  conf.set("spark.sql.autoBroadcastJoinThreshold", "1")
  
  val spark = SparkSession.builder
    .config(conf)
    .getOrCreate()

  val sc = spark.sparkContext
  val sqlContext = spark.sqlContext
  
  val cfg = new Config("conf/server.properties")
        
  val jdbc = null
  val ml = MLSample(sc,sqlContext)
  val ss = SStream(sc)
  val env = Env(system, cfg, fs, jdbc, ml, sc, sqlContext)
  
  val trans = Transform(spark)
  val mlgen = MLGenetor(spark)
  
  import scala.util._
  import env.sqlContext.implicits._
  
  import scala.reflect.runtime.{universe => ru}
  val typeMirror = ru.runtimeMirror(trans.getClass.getClassLoader)
  val instanceMirror = typeMirror.reflect(trans)
  val members = instanceMirror.symbol.typeSignature.members
  val transMems = members.filter(_.name.decoded.startsWith("trans"))
  val transNames = transMems.map(_.name.decoded).toSeq.sortBy(x=>x)
  
  val zip = spark.read
      //.schema(schema1)
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data2/superzip.csv")
  val superzipDS = zip.as("superzip")
  
  val features = superzipDS.schema.fields
  //基因长度
  val length = features.length * transNames.length
  println(s"----------$length")
   
  //迭代次数，最大子代数
  val MAX_GENERATION = 5
   
  //种群大小
  val POPULATION_SIZE = 5
   
  //变异概率
  val MUTATION_RATE = 0.1 
   
  /**
   * 染色体
   * sequence 基因序列，表示一个整数的二进制形式，从最小位开始写
   *          比如10的二进制是 00 0000 1010，十位，
   *          则sequence表示为 0101 0000 00
   * fitness  适应度值
   */
  case class Chromosome(sequence: Array[Int], var fitness: Double)
   
  /**
   * 适应度函数，计算 (x-512)*(x-512)-10 的值
   */
  def CalFitnessOne(sequence: Array[Int]) = {
    //将二进制转为十进制
    val x = (0.0 /: (0 until sequence.length)){
        (acc, elem) =>  acc + sequence(elem) * Math.pow(2.0, elem)
    }
    //代入函数得出适应度值
    (x - 512) * (x - 512) - 10
  }
   
  /**
   * 适应度函数2，计算 (x-512)*(x-256)-10 的值
   */
  //sequence sample: array with 50 (0/1)
  def CalFitnessTwo(sequence: Array[Int]) = {
    /*
    //将二进制转为十进制
    val x = (0.0 /: (0 until sequence.length)){
        (acc, elem) =>  acc + sequence(elem) * Math.pow(2.0, elem)
    }
    */
    
    var transDS = dataTransformProcess(sequence)
    
    //transDS.printSchema()
    //transDS.show()
    
    //代入函数得出适应度值
    val mseAvg = mlgen.decisionTreeMSE(transDS)
    val mseAvg2 = mlgen.decisionPipline(transDS)
    println(s"Root Mean Squared Error (RMSE) on data set = $mseAvg, $mseAvg2")
    val mse = (mseAvg+mseAvg2)/2
    println(mse)
    
    mse
    
  }
  
  def dataTransformProcess(sequence: Array[Int]) = {
      var transDS = superzipDS
      var handler = Seq[Tuple2[String, Seq[Int]]]()
    
      for(x <- transNames.zipWithIndex){
        var transIdx = Seq[Int]()
        for(y <- 0 to features.length-1){
          if(sequence(x._2*10 + y) > 0){
            transIdx :+= y
          }
        }
        handler :+= (x._1, transIdx)
      }
      
      handler.foreach(x => {
        val methodx = ru.typeOf[Transform].declaration(ru.newTermName(x._1)).asMethod
        transDS = instanceMirror.reflectMethod(methodx)(transDS, x._2).asInstanceOf[Dataset[Row]]
      })
      
      transDS
  }
   
  import scala.collection.mutable.PriorityQueue
  //优先级队列中元素比较的implicit ordering
  implicit object ChromosomeOrd extends Ordering[Chromosome] {
    def compare(x: Chromosome, y: Chromosome) = y.fitness.compare(x.fitness)
  }
   
  import java.util.Random
  val random = new Random
   
  /**
   * 初始化种群
   */
  def initPopulation(calFitness: Array[Int] => Double) = {
    val population = new PriorityQueue[Chromosome]
    for(i <- 0 until POPULATION_SIZE) {
      val sequence = Array.fill(length)(0).map(x => x ^ random.nextInt(2))
      population += Chromosome(sequence, calFitness(sequence))
    }
    population
  }
   
  /**
   * 简单的选择操作，保留优先级队列中前一半的基因
   */
  def selectChromosome(population: PriorityQueue[Chromosome]) = {
    val minList = (for(i <- 0 until POPULATION_SIZE / 2) 
            yield population dequeue).toList
    (new PriorityQueue[Chromosome] /: minList){
        (acc, elem) => acc += elem
    }
  }
   
  /**
   * 交叉变异操作
   */
  def CrossOver_Mutation(population: PriorityQueue[Chromosome],
            calFitness: Array[Int] => Double) = {
    //随机获取配偶的位置
    //po1为当前染色体的位置
    def getSpouse(po1: Int, population_size: Int): Int = {
      val spouse = random nextInt population_size
      if(spouse == po1) getSpouse(po1, population_size)
      else spouse
    }
     
    val populaSize = POPULATION_SIZE / 2
    val poList = population.toList
    val tempQueue = new PriorityQueue[Chromosome]
    for(i <- 0 until populaSize){
      val (seq1, seq2) = CrossOver(poList(i), 
                       poList(getSpouse(i, populaSize)),
                       calFitness)
      tempQueue += seq1
      tempQueue += seq2
    }
    tempQueue map (Mutation(_, calFitness))
  }
   
  /**
   * 交叉两个染色体，产生两个子代
   */
  def CrossOver(chromOne: Chromosome, chromTwo: Chromosome,
               calFitness: Array[Int] => Double) = {
    val position = random nextInt length - 1
    val seqOne =
      chromOne.sequence.take(position + 1) ++ 
      chromTwo.sequence.takeRight(length - position)
       
    val seqTwo =
      chromTwo.sequence.take(position) ++ 
      chromOne.sequence.takeRight(length - position)
       
    (Chromosome(seqOne, calFitness(seqOne)), 
        Chromosome(seqTwo, calFitness(seqTwo)))
  }
     
   /**
    * 染色体变异 
   */
   def Mutation(chrom: Chromosome, 
                calFitness: Array[Int] => Double) =
     //首先满足变异概率
     if(random.nextDouble > MUTATION_RATE){
          var seq = chrom.sequence
          val po = random nextInt length
          seq(po) = seq(po) ^ 1
          //若变异后适应值比原来大则不变异
          if(calFitness(seq) > calFitness(chrom.sequence))
            seq = chrom.sequence
          Chromosome(seq, calFitness(seq))
     } else chrom
     
   
  def main(args: Array[String]): Unit = {
    var population =  initPopulation(CalFitnessTwo)
    //var population =  initPopulation(CalFitnessOne)
    var smallest = population.max.fitness
    var temp = 0.0
    for(i <- 0 until MAX_GENERATION){
      population = selectChromosome(population)
      //population = CrossOver_Mutation(population, CalFitnessOne)
      population = CrossOver_Mutation(population, CalFitnessTwo)
      temp = population.max.fitness
      if(temp < smallest) smallest = temp
    }
    println(s"函数极值为 $smallest")
  }
 
}