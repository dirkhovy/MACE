'''
compare the annotator proficiency to the model's confidence
correlation.py <proficiency file> <confidence file>
'''

import sys
import scipy as sp
import scipy.stats as sps
from collections import defaultdict

proficiency = sp.array(map(float, open(sys.argv[1], 'rU').readlines()[0].strip().split()))
confidence = sp.array(map(float, open(sys.argv[2], 'rU').readlines()[0].strip().split()))

print sps.pearsonr(proficiency, confidence)