all: build submit

jar = sbdgenerator-assembly-0.1-SNAPSHOT.jar
args += --driver-java-options "-Dlog4j.configuration=file:conf/log4j.properties"
args += target/scala-2.11/$(jar)

build:
	sbt assembly -Dsbt.log.noformat=true

clean:
	rm -r target
	
submit_cdh:
	spark-submit --master yarn-client --class cdn.CDNSampleRemote $(args)

submit:
	spark-submit --class Main $(args)
	
stream:
	spark-submit --class StreamMain $(args)

dataload:
	spark-submit --class DataMassage $(args)

start:
	daemonize -l server.lock -p server.pid -e server.err -o server.out -c `pwd` \
		`which spark-submit` --class Main $(args)

stop:
	kill `cat server.pid` || true

restart:
	make stop
	sleep 1
	make start

	
sync: 
	cp -r Makefile conf data data2 target/scala-2.11/$(jar) /Users/admin/1_unix_server/sbdgenerator/
	
upload_data:
	hadoop fs -put -f /opt/sbdgenerator/data  hdfs://10.141.212.21:9000/user/root/
	hadoop fs -put -f /opt/sbdgenerator/data2  hdfs://10.141.212.21:9000/user/root/
	
submit_yarn:
	spark-submit --master yarn --class Main $(jar)
	
submit_yarn_damon:
	nohup spark-submit --master yarn --class Main $(jar) &
	
submit_monitor:
	tail -f nohup.out

	
	