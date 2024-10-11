import re

def findnumbers(string): return [float(x) for x in re.findall(r'-?\d+\.?\d*', string)]
