/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package streaming

import akka.actor._
import akka.io.IO
import akka.pattern._
import akka.routing._
import akka.util.Timeout
import org.apache.hadoop.fs._
import org.apache.hadoop.conf._
import org.apache.avro._
import org.apache.avro.file._
import org.apache.avro.reflect._
import org.apache.hadoop.fs._
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


import org.apache.spark.SparkConf
import org.apache.spark.mllib.stat.test.{BinarySample, StreamingTest}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.util.Utils

/**
 * Perform streaming testing using Welch's 2-sample t-test on a stream of data, where the data
 * stream arrives as text files in a directory. Stops when the two groups are statistically
 * significant (p-value < 0.05) or after a user-specified timeout in number of batches is exceeded.
 *
 * The rows of the text files must be in the form `Boolean, Double`. For example:
 *   false, -3.92
 *   true, 99.32
 *
 * Usage:
 *   StreamingTestExample <dataDir> <batchDuration> <numBatchesTimeout>
 *
 * To run on your local machine using the directory `dataDir` with 5 seconds between each batch and
 * a timeout after 100 insignificant batches, call:
 *    $ bin/run-example mllib.StreamingTestExample dataDir 5 100
 *
 * As you add text files to `dataDir` the significance test wil continually update every
 * `batchDuration` seconds until the test becomes significant (p-value < 0.05) or the number of
 * batches processed exceeds `numBatchesTimeout`.
 */
object StreamingTestExample {
  
  def initFileSystem(sc: SparkContext, props:Map[String,String]) = {
        val conf = sc.hadoopConfiguration
        props.foreach(Function.tupled(conf.set))
        FileSystem.enableSymlinks()
        FileSystem.get(conf)
    }

  def main(args: Array[String]) {
    val dataDir = "data_stream" //args(0)
    val batchDuration = Seconds(5) //Seconds(args(1).toLong)
    val numBatchesTimeout = 100//args(2).toInt

    val fs = FileSystem.get(new Configuration)
    val conf = new SparkConf().setMaster("local").setAppName("StreamingTestExample")
    val ssc = new StreamingContext(conf, batchDuration)
    ssc.checkpoint {
      val dir = fs.mkdirs(new Path("temp"))//Utils.createTempDir()
      dir.toString
    }

    // $example on$
    val data = ssc.textFileStream(dataDir).map(line => {
      println("-------content--------")
      println(line)
      line.split(",") match {
        case Array(label, value) => BinarySample(label.toBoolean, value.toDouble)
      }
    })
    
    data.foreachRDD(rdd=>{
        val spark = SparkSession.builder.config(rdd.sparkContext.getConf).getOrCreate()
        import spark.implicits._
        val ddata = rdd.toDF("label", "value")
        ddata.show()
    })

    val streamingTest = new StreamingTest()
      .setPeacePeriod(0)
      .setWindowSize(0)
      .setTestMethod("welch")

    val out = streamingTest.registerStream(data)
    out.print()
    // $example off$

    // Stop processing if test becomes significant or we time out
    var timeoutCounter = numBatchesTimeout
    out.foreachRDD { rdd =>
      timeoutCounter -= 1
      val anySignificant = rdd.map(_.pValue < 0.05).fold(false)(_ || _)
      if (timeoutCounter == 0 || anySignificant) rdd.context.stop()
    }

    ssc.start()
    ssc.awaitTermination()
  }
}
