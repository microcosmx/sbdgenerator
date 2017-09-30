

import org.apache.spark.SparkContext._
import org.apache.spark.streaming.{Seconds, StreamingContext}
// scalastyle:off
/** Analyses a streaming dataset of web page views. This class demonstrates several types of
  * operators available in Spark streaming.
  *
  * This should be used in tandem with PageViewStream.scala. Example:
  * To run the generator
  * `$ bin/run-example org.apache.spark.examples.streaming.clickstream.PageViewGenerator 44444 10`
  * To process the generated stream
  * `$ bin/run-example \
  *    org.apache.spark.examples.streaming.clickstream.PageViewStream errorRatePerZipCode localhost 44444`
  */
// scalastyle:on
object RestServiceStream {
  def main(args: Array[String]) {
    
    val metric = "pageCounts"
    val host = "localhost"
    val port = 44444

    // Create the context
    val ssc = new StreamingContext("local[2]", "PageViewStream", Seconds(1),
      System.getenv("SPARK_HOME"), StreamingContext.jarOfClass(this.getClass).toSeq)

    // Create a ReceiverInputDStream on target host:port and convert each line to a PageView
    val pageViews = ssc.socketTextStream(host, port)
                       .flatMap(_.split("\n"))
                       .map(PageView.fromString(_))

    // Return a count of views per URL seen in each batch
    val pageCounts = pageViews.map(view => view.url).countByValue()

    // Return a sliding window of page views per URL in the last ten seconds
    val slidingPageCounts = pageViews.map(view => view.url)
                                     .countByValueAndWindow(Seconds(10), Seconds(2))


    // Return the rate of error pages (a non 200 status) in each zip code over the last 30 seconds
    val statusesPerZipCode = pageViews.window(Seconds(30), Seconds(2))
                                      .map(view => ((view.zipCode, view.status)))
                                      .groupByKey()
    val errorRatePerZipCode = statusesPerZipCode.map{
      case(zip, statuses) =>
        val normalCount = statuses.filter(_ == 200).size
        val errorCount = statuses.size - normalCount
        val errorRatio = errorCount.toFloat / statuses.size
        if (errorRatio > 0.05) {
          "%s: **%s**".format(zip, errorRatio)
        } else {
          "%s: %s".format(zip, errorRatio)
        }
    }

    // Return the number unique users in last 15 seconds
    val activeUserCount = pageViews.window(Seconds(15), Seconds(2))
                                   .map(view => (view.userID, 1))
                                   .groupByKey()
                                   .count()
                                   .map("Unique active users: " + _)

    // An external dataset we want to join to this stream
    val userList = ssc.sparkContext.parallelize(
       Map(1 -> "Patrick Wendell", 2->"Reynold Xin", 3->"Matei Zaharia").toSeq)

    metric match {
      case "pageCounts" => {
        System.out.println("====================")
        pageCounts.print()
      }
      case "slidingPageCounts" => slidingPageCounts.print()
      case "errorRatePerZipCode" => errorRatePerZipCode.print()
      case "activeUserCount" => activeUserCount.print()
      case "popularUsersSeen" =>
        // Look for users in our existing dataset and print it out if we have a match
        pageViews.map(view => (view.userID, 1))
          .foreachRDD((rdd, time) => rdd.join(userList)
            .map(_._2._2)
            .take(10)
            .foreach(u => println("Saw user %s at time %s".format(u, time))))
      case _ => println("Invalid metric entered: " + metric)
    }

    ssc.start()
    ssc.awaitTermination()
  }
}