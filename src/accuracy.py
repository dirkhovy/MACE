'''
accuracy.py <key> <prediction>

'''
import sys

key = map(str.strip, open(sys.argv[1], "rU").readlines())
pred = map(str.strip, open(sys.argv[2], "rU").readlines())

total = 0.0
correct = 0.0
for i, prediction in enumerate(pred):
    truth = key[i]
    if truth not in ["no consensus", "withdraw"]:
        total += 1.0
	if truth == prediction:
	   correct += 1.0
 
print correct/total
