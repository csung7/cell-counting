#!/usr/bin/env python
# encoding: utf-8

from itertools import groupby
from operator import itemgetter
import numpy as np
import sys

def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.rstrip().split(separator, 1)

def main(separator='\t'):
    # input comes from STDIN (standard input)
    data = read_mapper_output(sys.stdin, separator=separator)
    # data: x,y,z /t vol# /t class# /t dist-err /t proxi
    # groupby groups multiple point-dist errors pairs by point,
    # and creates an iterator that returns consecutive keys and their group:
    #   c_point - string containing x,y,z (the key)
    #   group - c_vol, c_class, c_dist items
    for c_point, group in groupby(data, itemgetter(0)):
        loop_p = 0
        loop_n = 0
        avg_p_dist = 0.0
        avg_n_dist = 0.0
        avg_dist = 0.0
        for c_point, c_contents in group:
            c_vol, c_class, c_dist = c_contents.rstrip().split('\t')
            if(c_class == "0"):
                avg_n_dist = avg_n_dist + np.float(c_dist)
                loop_n = loop_n + 1
            else:
                avg_p_dist = avg_p_dist + np.float(c_dist)
                loop_p = loop_p + 1
        avg_dist = (avg_p_dist/loop_p) - (avg_n_dist/loop_n)
        print "%s%s\t%0.4f" % (c_point, separator, np.round(avg_dist,4))

if __name__ == "__main__":
    main()
