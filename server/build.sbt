scalaVersion := "2.11.8"
name := "sbdgenerator"

libraryDependencies ++= Seq(
    "io.spray" %% "spray-can"       % "1.3.3",
    "io.spray" %% "spray-http"      % "1.3.3",
    "io.spray" %% "spray-routing"   % "1.3.3",
    "io.spray" %% "spray-client"    % "1.3.3",
    "io.spray" %% "spray-json"      % "1.3.2",
    
    "log4j" % "log4j" % "1.2.17",

    "org.gnieh" %% "diffson" % "1.1.0",

    "org.apache.kafka" %% "kafka" % "0.8.2.1",
    
    "org.apache.spark" %% "spark-core"      % "2.0.0" % "provided",
    "org.apache.spark" %% "spark-sql"       % "2.0.0" % "provided",
    "org.apache.spark" %% "spark-hive"      % "2.0.0" % "provided",
    "org.apache.spark" %% "spark-mllib"     % "2.0.0" % "provided",
    "org.apache.spark" %% "spark-streaming" % "2.0.0" % "provided",
  	//"org.apache.spark" %% "spark-streaming-kafka" % "2.0.0" % "provided",
  	//"org.apache.spark" %% "spark-streaming-flume" % "2.0.0" % "provided",
  	"org.apache.spark" %% "spark-repl" % "2.0.0" % "provided",
  	
  	//"eu.piotrbuda" %% "scalawebsocket" % "0.1.1",
    
    "org.slf4j" % "slf4j-api" % "1.7.5",
    "org.slf4j" % "slf4j-log4j12" % "1.7.5",

    "org.scalaz" %% "scalaz-core" % "7.2.2",
  
    "com.jsuereth" %% "scala-arm" % "1.4",

    "org.scalatest" %% "scalatest" % "2.2.6" % "test",
    "org.scalacheck" %% "scalacheck" % "1.13.0" % "test",
    
    "com.typesafe.akka" %% "akka-actor" % "2.3.15",
    "com.typesafe.akka" %% "akka-testkit" % "2.4.4" % "test"
	
)

test in assembly := {}

fork := true

run in Compile <<= Defaults.runTask(fullClasspath in Compile, mainClass in (Compile, run), runner in (Compile, run)) 
