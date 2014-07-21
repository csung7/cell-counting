#!/bin/sh
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
