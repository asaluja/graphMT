#!/usr/bin/python -tt

'''
File: analyze_pt.py
Date: January 31, 2014
Description: prints out statistics of a standard phrase table. 
'''

import sys, commands, string, gzip

filehandle = gzip.open(sys.argv[1])
numPhrasePairs = 0
srcPhraseDict = {}
tgtPhraseDict = {}
maxPLSrc = 0
maxPLTgt = 0
for line in filehandle:
    numPhrasePairs += 1
    elements = line.strip().split(' ||| ')
    srcPhr = elements[0]
    tgtPhr = elements[1]
    if len(srcPhr.split()) > maxPLSrc:
        maxPLSrc = len(srcPhr.split())
    if len(tgtPhr.split()) > maxPLTgt:
        maxPLTgt = len(tgtPhr.split())
    if srcPhr not in srcPhraseDict:
        srcPhraseDict[srcPhr] = 1
    if tgtPhr not in tgtPhraseDict:
        tgtPhraseDict[tgtPhr] = 1

filehandle.close()
print "Number of phrase pairs: %d"%(numPhrasePairs)
print "Number of source side phrases: %d"%(len(srcPhraseDict))
print "Number of target side phrases: %d"%(len(tgtPhraseDict))
print "Longest source phrase: %d"%maxPLSrc
print "Longest target phrase: %d"%(maxPLTgt)
    
    
    
    
