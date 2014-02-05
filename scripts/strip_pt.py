#!/usr/bin/python -tt

'''
File:
Date: February 2, 2014
Description:
'''

import sys, commands, string

for line in sys.stdin:
    elements = line.strip().split(' ||| ')
    print ' ||| '.join(elements[1:])
