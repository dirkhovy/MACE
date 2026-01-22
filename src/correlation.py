'''
compare the annotator proficiency to the model's confidence
correlation.py <proficiency file> <confidence file>
'''

import sys
import scipy as sp
import scipy.stats as sps
from collections import defaultdict

proficiency = sp.array([float(x) for x in open(sys.argv[1], 'r').readlines()[0].strip().split()])
confidence = sp.array([float(x) for x in open(sys.argv[2], 'r').readlines()[0].strip().split()])

print(sps.pearsonr(proficiency, confidence))
