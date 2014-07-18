#!/bin/sh
################ tn mapper ################
T="$(date +%s%N)"
hadoop jar $HADOOP_HOME/contrib/streaming/hadoop-*streaming*.jar -D mapred.reduce.tasks=0 -files hdfs:/user/hduser/cdata/training05_n.cel,hdfs:/user/hduser/cdata/training05_n.vol,hdfs:/user/hduser/cdata/training08b_n.cel,hdfs:/user/hduser/cdata/training08b_n.vol,hdfs:/user/hduser/cdata/training10c_n.cel,hdfs:/user/hduser/cdata/training10c_n.vol,/home/hduser/hadoop-samples/cell-counting/src/tnmapper.py -mapper tnmapper.py -input /user/hduser/ccount/tnindex.txt -output /user/hduser/ccount/tnmout
T="$(($(date +%s%N)-T))"
# Seconds
S="$((T/1000000000))"
# Milliseconds
M="$((T/1000000))"
echo "Time in nanoseconds: ${T}"
printf "Pretty format: %02d:%02d:%02d:%02d.%03d\n" "$((S/86400))" "$((S/3600%24))" "$((S/60%60))" "$((S%60))" "${M}"
################ ts mapper ################
T="$(date +%s%N)"
hadoop jar $HADOOP_HOME/contrib/streaming/hadoop-*streaming*.jar -files hdfs:/user/hduser/cdata/training06_n.cel,hdfs:/user/hduser/cdata/training06_n.vol,/home/hduser/hadoop-samples/cell-counting/src/tsmapper.py,/home/hduser/hadoop-samples/cell-counting/src/tsreducer.py -mapper tsmapper.py -reducer tsreducer.py -input /user/hduser/ccount/tnmout/part-* -output /user/hduser/ccount/tsout
T="$(($(date +%s%N)-T))"
# Seconds
S="$((T/1000000000))"
# Milliseconds
M="$((T/1000000))"
echo "Time in nanoseconds: ${T}"
printf "Pretty format: %02d:%02d:%02d:%02d.%03d\n" "$((S/86400))" "$((S/3600%24))" "$((S/60%60))" "$((S%60))" "${M}"
#-file hdfs:/user/hduser/cdata/training05_n.cel \
#-file hdfs:/user/hduser/cdata/training05_n.vol \
#-file hdfs:/user/hduser/cdata/training08b_n.cel \
#-file hdfs:/user/hduser/cdata/training08b_n.vol \
#-file hdfs:/user/hduser/cdata/training10c_n.cel \
#-file hdfs:/user/hduser/cdata/training10c_n.vol \
#-file /home/hduser/hadoop-samples/cell-counting/src/tnmapper.py \

