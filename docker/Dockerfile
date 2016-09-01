FROM bigdatauniversity/spark2:latest

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 

RUN mkdir -p /opt/sbdgenerator

ENV RUN_JAR sbdgenerator-assembly-0.1-SNAPSHOT.jar

COPY ../server/target/scala-2.11/$RUN_JAR /opt/sbdgenerator
ADD ../server/conf /opt/sbdgenerator
ADD ../server/lib /opt/sbdgenerator

ENV SBD_HOME /opt/sbdgenerator

WORKDIR $SBD_HOME

CMD spark-submit --class Main --driver-java-options "-Dlog4j.configuration=file:conf/log4j.properties" $RUN_JAR
