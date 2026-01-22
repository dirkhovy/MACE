'''
get_majority_voting.py <file> [<controls>]

'''
import sys
import random

controls = None
if len(sys.argv) == 3:
    controls = [line.strip() for line in open(sys.argv[2], "r").readlines()]

pred = [line.strip() for line in open(sys.argv[1], "r").readlines()]

for i, line in enumerate(pred):
    if controls and controls[i] != "":
        print(controls[i])
    else:
        items = line.replace(',',' ').split()
        counts = [ (items.count(x), x) for x in set(items) ]
        highest = max(counts)[0]
        candidates = [count[1] for count in counts if count[0] == highest]    
        truth = random.choice(candidates)
        print(truth)
