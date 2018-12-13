f = open('da', 'r')

lines = f.readlines() 
lines2 = []
for line in lines:
    if " loss: " in line:
        lines2.append(line)

vals = []
for line in lines2:
    sections = line.split('loss')
    v = sections[1].split()
    vals.append(float(v[1]))

import ipdb; ipdb.set_trace()
