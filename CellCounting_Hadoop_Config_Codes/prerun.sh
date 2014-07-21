#!/bin/sh
hadoop fs -mkdir /user/hduser/ccount
hadoop fs -put ./tnindex.txt /user/hduser/ccount/tnindex.txt
hadoop fs -mkdir /user/hduser/cdata
hadoop fs -put ./../n_data/training05_n.cel /user/hduser/cdata/training05_n.cel
hadoop fs -put ./../n_data/training08b_n.cel /user/hduser/cdata/training08b_n.cel
hadoop fs -put ./../n_data/training10c_n.cel /user/hduser/cdata/training10c_n.cel
hadoop fs -put ./../n_data/training06_n.cel /user/hduser/cdata/training06_n.cel
hadoop fs -put ./../n_data/training05_n.vol /user/hduser/cdata/training05_n.vol
hadoop fs -put ./../n_data/training08b_n.vol /user/hduser/cdata/training08b_n.vol
hadoop fs -put ./../n_data/training10c_n.vol /user/hduser/cdata/training10c_n.vol
hadoop fs -put ./../n_data/training06_n.vol /user/hduser/cdata/training06_n.vol
