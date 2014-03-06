#!/usr/bin/python 

'''
File:
Date: February 21, 2014
Description: 
'''

import sys, string, math
import cdec.configobj
import cdec.sa

MAXSCORE=99

def computeLexicalScores(model_loc, src_phrases, tgt_phrases):
    tt = cdec.sa.BiLex(from_binary=model_loc)
    lexScores = []
    for idx,src in enumerate(src_phrases):
        src_words = src.split()
        tgt_words = tgt_phrases[idx].split()
        e_given_f = maxLexEgivenF(src_words, tgt_words, tt)
        f_given_e = maxLexFgivenE(src_words, tgt_words, tt)
        lexScores.append((e_given_f, f_given_e))
    return lexScores

def maxLexEgivenF(fwords, ewords, ttable):
    fwords.append('NULL')
    maxOffScore = 0.0
    for e in ewords:
        maxScore = max(ttable.get_score(f, e, 0) for f in fwords)
        maxOffScore += -math.log10(maxScore) if maxScore > 0 else MAXSCORE
    return math.pow(10, -maxOffScore)

def maxLexFgivenE(fwords, ewords, ttable):
    ewords.append('NULL')
    maxOffScore = 0.0
    for f in fwords:
        maxScore = max(ttable.get_score(f, e, 1) for e in ewords)
        maxOffScore += -math.log10(maxScore) if maxScore > 0 else MAXSCORE
    return math.pow(10, -maxOffScore)

def main():
    src_pps = []
    tgt_pps = []
    for line in sys.stdin:
        elements = line.strip().split(' ||| ')
        src_pps.append(elements[0])
        tgt_pps.append(elements[1])
    lexScores = computeLexicalScores(sys.argv[1], src_pps, tgt_pps)
    for fwdScore, bwdScore in lexScores:
        print "Forward score: %.3f; Backward score: %.3f"%(fwdScore, bwdScore)

if __name__ == "__main__":
    main()
