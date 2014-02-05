#!/usr/bin/python -tt

'''
File: filter_pt.py
Date: February 2, 2014
Description: this script takes as input in stdin a grammar extracted
from the parallel training data (i.e., not per-sentence grammar), and
two options: whether to filter singletons, and whether to sort by 
prob(target phrase | source phrase) and take the top N only. 
'''

import sys, commands, string, getopt

def main():
    optDict = {}
    (opts, args) = getopt.getopt(sys.argv[1:], 'f:s')
    for opt in opts:
        if opt[0] == '-s': #singleton removal
            optDict["noSingletons"] = 1
        elif opt[0] == '-f': #filter rules
            optDict["filterRules"] = int(opt[1])    
    for line in sys.stdin:
        elements = line.strip().split(' ||| ')
        features = elements[3]
        featureDict = dict(item.split("=") for item in features.split())
        if "noSingletons" in optDict:
            if featureDict["IsSingletonFE"] != "1.0":
                print line.strip()
    

if __name__ == "__main__":
    main()
