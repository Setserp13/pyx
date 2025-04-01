import re
from pyx.collectionsx import List

#def findnumbers(string): return [float(x) for x in re.findall(r'-?\d+\.?\d*', string)]
def findnumbers(string): return [float(x) if '.' in x else int(x) for x in re.findall(r'-?\d+\.?\d*', string)]

def strpdict(obj, sep=[';', ':']):
	result = {}
	List.items = [] if obj == '' else obj.split(sep[0])
	for item in List.items:
		key, value = item.split(sep[1])
		result[key.strip()] = value.strip()
	return result

def strfdict(obj, sep=[';', ':']): return sep[0].join([f'{k}{sep[1]}{obj[k]}' for k in obj])

def to_intervals(string):
	labels = [rex.findnumbers(x) for x in string.split('\n')]
	for i, x in enumerate(text.split('\n')):
		#print(i)
		if i < len(labels) - 1:
			if labels[i][0] == labels[i][1]:
				labels[i][1] = labels[i+1][0]
