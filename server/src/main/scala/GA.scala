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

  
case class GA(
    spark: SparkSession,
    env: Env,
    trans: Transform,
    mlgen: MLGenetor,
    mlreg: MLRegression)
{
  
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
  val featureNames = features.map(_.name).toSeq
  
  val actions = spark.read
      //.schema(schema1)
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data2/action.csv")
  val actionsDS = actions.as("actions")
  actionsDS.show
  
  val actionsList = actionsDS.collect.map { row => 
      val target1 = row.getAs[String]("target")
      val action1 = row.getAs[String]("action")
      val object1 = row.getAs[String]("object")
      if(target1!="all" && action1!="related_to" && object1!="any"){
        ("level1", (target1.split(" ").toSeq, action1.split(" ").toSeq, object1.split(" ").toSeq))
      }else{
        ("level3", (
            if(target1.contains("all")) Seq()/*featureNames.toSeq*/ else target1.split(" ").toSeq,
            action1.split(" ").toSeq, 
            if(object1.contains("any")) Seq()/*featureNames.toSeq*/ else object1.split(" ").toSeq
        ))
      }
  }
  val actionLevel1 = actionsList.filter(_._1 == "level1").map(_._2).toSeq
  val actionLevel3 = actionsList.filter(_._1 == "level3").map(_._2).toSeq
            
            
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
    val targets_ = actionLevel3.flatMap(_._1).intersect(featureNames).toSeq
    val objects_ = actionLevel3.flatMap(_._3).intersect(featureNames).diff(targets_).toSeq
    val actions_ = actionLevel3.flatMap(_._2).toSeq
    
    val targets_left = featureNames.diff(targets_)
    val objects_left = featureNames.diff(objects_)
    
    val actionLevel3Process = Seq(
        ((targets_, actions_, objects_), 4),
        ((targets_, actions_, objects_left), 3),
        ((targets_left.diff(objects_), actions_, objects_), 2),
        ((targets_left, actions_, objects_left), 1)
    )
    
    val result = actionLevel3Process.map(actps=>{
        val result1 = mlgen.decisionTreeMSE(transDS, actps._1._1, actps._1._3)
        val result2 = mlgen.decisionPipline(transDS, actps._1._1, actps._1._3)
        val result3 = mlgen.decision_Multilayer_perceptron_classifier(transDS, actps._1._1, actps._1._3)
        val result4 = mlreg.decision_randomforest(transDS, actps._1._1, actps._1._3)
        val result5 = mlreg.decision_Gradient_boosted_tree(transDS, actps._1._1, actps._1._3)
        val minresult = Seq(result1, result2, result3, result4, result5).sortBy(x => x).head
        println(minresult, result1, result2, result3, result4, result5)
        val retVal = if(minresult < 0.5) minresult else -1.0
        (retVal, actps._2)
    }).toSeq
    println(result)
    
    val validResult = result.filter(_._1 > 0)
    println(validResult)
    
    val targetValue = validResult.map(_._1).sum / math.pow(validResult.map(_._2).sum, 1.5)
    println(s"-------target value is-------${targetValue}----------")
    
    targetValue
    
  }
  
  def dataTransformProcess(sequence: Array[Int]) = {
      var transDS = superzipDS//.filter(x => {x.getInt(0) > 1100}).sort(features(0).name)
            
      transDS.printSchema()
      transDS.show()
      //data transform
      actionLevel1.foreach { x=>
          if(x._2.head == "filter"){
              transDS = transDS.filter(s"${x._1.head} != ${x._3.head}") 
          }else{
              //TODO
          }
      }
      transDS.show
      
      val length = features.length * transNames.length
      val sequence = Array.fill(length)(0).map(x => x ^ Random.nextInt(2))
      
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
   
  /**
   * 初始化种群
   */
  def initPopulation(calFitness: Array[Int] => Double) = {
    val population = new PriorityQueue[Chromosome]
    for(i <- 0 until POPULATION_SIZE) {
      val sequence = Array.fill(length)(0).map(x => x ^ Random.nextInt(2))
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
      val spouse = Random nextInt population_size
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
    val position = Random nextInt length - 1
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
     if(Random.nextDouble > MUTATION_RATE){
          var seq = chrom.sequence
          val po = Random nextInt length
          seq(po) = seq(po) ^ 1
          //若变异后适应值比原来大则不变异
          if(calFitness(seq) > calFitness(chrom.sequence))
            seq = chrom.sequence
          Chromosome(seq, calFitness(seq))
     } else chrom
     
   
//  def main(args: Array[String]): Unit = {
//    var population =  initPopulation(CalFitnessTwo)
//    //var population =  initPopulation(CalFitnessOne)
//    var smallest = population.max.fitness
//    var temp = 0.0
//    for(i <- 0 until MAX_GENERATION){
//      population = selectChromosome(population)
//      //population = CrossOver_Mutation(population, CalFitnessOne)
//      population = CrossOver_Mutation(population, CalFitnessTwo)
//      temp = population.max.fitness
//      if(temp < smallest) smallest = temp
//    }
//    println(s"函数极值为 $smallest")
//  }
 
}