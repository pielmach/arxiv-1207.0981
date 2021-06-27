#!/usr/bin/env python

"""
author:       Michael Prim
contact:      michael.prim@kit.edu
date:         2012-05-31
version:      1.0
description:  Generates a testdata sample and header_dict example to be used in CAT
"""

import csv
import math, random

def main():
    # write header_dict
    filename = open('testdata_header_dict.txt', 'w')
    filename.write("header_dict.__setitem__('var-a', r'$a = U(0,1)$')\n")
    filename.write("header_dict.__setitem__('var-b', r'$b = G(0,1)$')\n")
    filename.write("header_dict.__setitem__('var-c', r'$c = a \\times b$')\n")
    filename.write("header_dict.__setitem__('var-d', r'$d = U(a,1)$')\n")
    filename.write("header_dict.__setitem__('var-e', r'$e = r \\times \\cos \\phi$')\n")
    filename.write("header_dict.__setitem__('var-f', r'$f = r \\times \\sin \\phi$')\n")

    # write input data and header
    writer = csv.writer(open('testdata.csv', 'wb'), delimiter = ';', quotechar = '|')
    header = ['var-a', 'var-b', 'var-c', 'var-d', 'var-e', 'var-f']
    writer.writerow(header)

    for i in xrange(0, 25000):
        a = random.uniform(0, 1)
        b = random.gauss(0, 1)
        c = a * b
        d = random.uniform(a, 1)
        radius = random.gauss(0.7, 0.15)
        phi = random.uniform(0, 2 * math.pi)
        e = radius * math.cos(phi)
        f = radius * math.sin(phi)
        writer.writerow([a, b, c, d, e, f])

    print 'To run CAT on the testdata simply call it with the following command:'
    print './cat_run_correlation_analysis.py -i testdata.csv -o testdata.pdf'

    print 'To additionally activate the TeX support, call it using:'
    print './cat_run_correlation_analysis.py -i testdata.csv -o testdata.pdf -t testdata_header_dict.txt'

    print 'In both cases you might want to activate the verbose mode using -v'
    print 'For help and more details about (other) options, call ./cat_run_correlation_analysis.py -h'

if __name__ == '__main__':
    main()
