cell-counting
=============

Cell counting algorithm for the Knife-edge scanning microscope brain atlas.


mapred.tasktracker.tasks.maximum

http://mit.edu/~mriap/hadoop/hadoop-0.13.1/docs/hadoop-default.html

EC2Instances.info
http://www.ec2instances.info/

http://www.hulu.com/watch/159320

https://console.aws.amazon.com/ec2/v2/home?region=us-east-1

Amazon EC2 Instance Types
http://aws.amazon.com/ec2/instance-types/

whirr launch-cluster --config hadoop-ec2.properties
cd ~/.whirr/hadoop-ec2
./hadoop-proxy.sh
export HADOOP_CONF_DIR=~/.whirr/hadoop-ec2
whirr destroy-cluster --config ~/hadoop-ec2.properties
http://54.242.212.25:50070

You can log into instances using the following ssh commands:
[hadoop-datanode+hadoop-tasktracker]: ssh -i /home/hduser/.ssh/id_rsa
-o "UserKnownHostsFile /dev/null" -o StrictHostKeyChecking=no
hduser@23.20.161.61

[hadoop-datanode+hadoop-tasktracker]: ssh -i /home/hduser/.ssh/id_rsa
-o "UserKnownHostsFile /dev/null" -o StrictHostKeyChecking=no
hduser@54.235.224.200

[hadoop-datanode+hadoop-tasktracker]: ssh -i /home/hduser/.ssh/id_rsa
-o "UserKnownHostsFile /dev/null" -o StrictHostKeyChecking=no
hduser@50.16.86.227

[hadoop-datanode+hadoop-tasktracker]: ssh -i /home/hduser/.ssh/id_rsa
-o "UserKnownHostsFile /dev/null" -o StrictHostKeyChecking=no
hduser@23.22.52.34

[hadoop-datanode+hadoop-tasktracker]: ssh -i /home/hduser/.ssh/id_rsa
-o "UserKnownHostsFile /dev/null" -o StrictHostKeyChecking=no
hduser@107.20.76.200
[hadoop-namenode+hadoop-jobtracker]: ssh -i /home/hduser/.ssh/id_rsa
-o "UserKnownHostsFile /dev/null" -o StrictHostKeyChecking=no
hduser@67.202.42.47

http://mit.edu/~mriap/hadoop/hadoop-0.13.1/docs/hadoop-default.html

when you change the configuration of whirr, change whirr.cluster-name.

export HADOOP_CONF_DIR=~/.whirr/hadoop-ec2-3

wget http://kesm.cs.tamu.edu:/hrun.sh

sudo apt-get update

sudo apt-get install python-setuptools python-dev build-essential
sudo apt-get install python-sklearn
Y
sudo easy_install pip
sudo pip install --upgrade pip
sudo apt-get install libpng-dev
sudo apt-get install zlib1g-dev libncurses5-dev
sudo apt-get install libfreetype6-dev
sudo pip uninstall matplotlib
sudo pip install matplotlib
wget http://kesm.cs.tamu.edu:/cell-counting-mapreduce.tar
tar -xvf cell-counting-mapreduce.tar
sudo mv cell* /usr/local/lib/python2.7/dist-packages/




Amazon EC2 Instance Types <http://aws.amazon.com/ec2/instance-types/>
M1 Small Instance – default*

1.7 GiB memory
1 EC2 Compute Unit (1 virtual core with 1 EC2 Compute Unit)
160 GB instance storage
32-bit or 64-bit platform
I/O Performance: Moderate
EBS-Optimized Available: No
API name: m1.small


default map 2, reduce 1
1M/1S
Training
Pretty format: 00:00:00:39.39473
Testing
Pretty format: 00:00:19:08.1148577

1M/2S
Training
Pretty format: 00:00:00:40.40822
Testing
Pretty format: 00:00:19:31.1171791

1M/3S
Training
Pretty format: 00:00:00:31.31622
Testing
Pretty format: 00:00:10:16.616838

1M/4S
Training
Pretty format: 00:00:00:34.34335
Testing
Pretty format: 00:00:11:50.710931


1M/5S
Training
Pretty format: 00:00:00:31.31567
Testing
Pretty format: 00:00:10:08.608435

1M/10S
Training
Pretty format: 00:00:00:36.36419
Testing
Pretty format: 00:00:10:11.611874



http://www.ec2instances.info/
Cluster Compute Quadruple Extra Large 23.00 GB 33.5 (2xIntel Xeon
X5570) 1690 GB (2x840 GB) 64-bit Very High 1 cc1.4xlarge $1.30 hourly
$1.61 hourly


hduser@ip-10-17-50-230:~$ echo $JAVA_HOME
/usr/lib/jvm/java-1.6.0-openjdk-amd64
hduser@ip-10-17-50-230:~$ echo $HADOOP_HOME
/usr/local/hadoop-1.1.1

Pretty format: 00:00:04:11.251533

m1/s10

training
m5
Pretty format: 00:00:00:22.22017
testing
m10/r5
Pretty format: 00:00:00:51.51616

training
m10
Pretty format: 00:00:00:15.15602
testing
m10/r10
Pretty format: 00:00:00:48.48779


training
m15
Pretty format: 00:00:00:19.19206
testing
m15/r10
Pretty format: 00:00:00:38.38545


training
m15
Pretty format: 00:00:00:18.18608
testing
m15/r15
Pretty format: 00:00:00:38.38722


training
m10
Pretty format: 00:00:00:19.19206
testing
m20/r10
Pretty format: 00:00:00:33.33885

training
m5
Pretty format: 00:00:00:15.15566
testing
m30/r10
Pretty format: 00:00:00:29.29964




1M/1S
Training
Pretty format: 00:00:00:18.18241
Testing
Pretty format: 00:00:02:42.162545

1M/3S
Training
Pretty format: 00:00:00:19.19262
Testing
Pretty format: 00:00:02:32.152600

1M/5S
Training
Pretty format: 00:00:00:19.19262
Testing
Pretty format: 00:00:02:32.152600


mapred.map.tasks 2 The default number of map tasks per job. Typically
set to a prime several times greater than number of available hosts.
Ignored when mapred.job.tracker is "local".


mapred.reduce.tasks 1 The default number of reduce tasks per job.
Typically set to a prime close to the number of available hosts.
Ignored when mapred.job.tracker is "local".


We used two different cluster environments.



1M/1S
Training
Map 1, Reduce 0
Pretty format: 00:00:00:19.19340
Testing
Map 1, Reduce 1
Pretty format: 00:00:04:34.274196

Training
Map 2, Reduce 0
Pretty format: 00:00:00:16.16902

Testing
Map 2, Reduce 2
Pretty format: 00:00:02:31.151268


Training
Map 5, Reduce 0
Pretty format: 00:00:00:16.16752

Testing
Map 5, Reduce 5
Pretty format: 00:00:01:17.77429


Training
Map 10, Reduce 0
Pretty format: 00:00:00:16.16628
Testing
Map 10, Reduce 5
Pretty format: 00:00:01:19.79493


Training
Map 10, Reduce 0
Pretty format: 00:00:00:16.16713
Testing
Map 10, Reduce 10
Pretty format: 00:00:01:27.87212

Training
Map 15, Reduce 0
Pretty format: 00:00:00:16.16806
Testing
Map 15, Reduce 1
Pretty format: 00:00:01:11.71096

Testing
Map 15, Reduce 3
Pretty format: 00:00:01:05.65290

Testing
Map 15, Reduce 5
Pretty format: 00:00:01:05.65089

Testing
Map 15, Reduce 10
Pretty format: 00:00:01:16.76794


Testing
Map 15, Reduce 15
Pretty format: 00:00:01:21.81301

Testing
Map 25, Reduce 5
Pretty format: 00:00:01:11.71998

Training
Map 25, Reduce 0
Pretty format: 00:00:00:16.16293


Testing
Map 25, Reduce 10
Pretty format: 00:00:01:17.77713


Training
Map 35, Reduce 0
Pretty format: 00:00:00:16.16325

Testing
Map 35, Reduce 10
Pretty format: 00:00:01:25.85855


Testing
Map 35, Reduce 25
Pretty format: 00:00:01:55.115676





1M/5S
Training
Map 1, Reduce 0
Pretty format: 00:00:00:18.18299
Testing
Map 1, Reduce 1
Pretty format: 00:00:04:55.295023

Training
Map 2, Reduce 0
Pretty format: 00:00:00:16.16830
Testing
Map 2, Reduce 2
Pretty format: 00:00:02:33.153985

Training
Map 5, Reduce 0
Pretty format: 00:00:00:15.15959
Testing
Map 5, Reduce 5
Pretty format: 00:00:01:14.74914

Training
Map 10, Reduce 0
Pretty format: 00:00:00:16.16525
Testing
Map 10, Reduce 5
Pretty format: 00:00:00:50.50361


Training
Map 10, Reduce 0
Pretty format: 00:00:00:16.16525
Testing
Map 10, Reduce 10
Pretty format: 00:00:00:51.51937

Testing
Map 15, Reduce 1
Pretty format: 00:00:00:42.42437

Testing
Map 15, Reduce 3
Pretty format: 00:00:00:42.42684

Testing
Map 15, Reduce 5
Pretty format: 00:00:00:41.41557

Training
Map 15, Reduce 0
Pretty format: 00:00:00:17.17247
Testing
Map 15, Reduce 10
Pretty format: 00:00:00:42.42890

Training
Map 15, Reduce 0
Pretty format: 00:00:00:15.15351
Testing
Map 15, Reduce 15
Pretty format: 00:00:00:49.49243

Testing
Map 25, Reduce 5
Pretty format: 00:00:00:47.47217

Training
Map 25, Reduce 0
Pretty format: 00:00:00:15.15722
Testing
Map 25, Reduce 10
Pretty format: 00:00:00:47.47594


Training
Map 35, Reduce 0
Pretty format: 00:00:00:15.15599
Testing
Map 35, Reduce 10
Pretty format: 00:00:00:54.54242


Testing
Map 35, Reduce 25
Pretty format: 00:00:01:07.67716
