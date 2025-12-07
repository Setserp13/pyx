import re
from pyx.collectionsx import List
import pyx.mat.mat as mat

#data can be bytes, string and so on...
def ljust(data, width, fill): return data + fill * max(0, width - len(data))	#pads up to an exact width
def lpad(data, align, fill): return ljust(data, mat.ceil(len(data), align), fill)	#pads until length is a multiple of align

def split_at(lst, *indices):
	if len(indices) == 0:
		return [lst]
	result = [lst[:indices[0]]]
	for i in range(1, len(indices)):
		result.append(lst[indices[i-1]:indices[i]])
	result.append(lst[indices[-1]:])
	return result

#RANGE NOTATION

def collapse(lst):	#Returns a string
	lst = sorted(lst)
	indices = [i for i in range(1, len(lst)) if lst[i] - lst[i-1] > 1]
	result = split_at(lst, *indices)
	result = [f'{x[0]}-{x[-1]}' if len(x) > 1 else str(x[0]) for x in result]
	return ','.join(result)

def expand(s):  # Returns a list of ints
	s = s.replace(' ', '')
	result = []
	for x in s.split(','):
		y = x.split('-')
		if len(y) == 1:
			result.append(int(y[0]))
		else:
			start, end = int(y[0]), int(y[1])
			result.extend(range(start, end + 1))
	return result

#RANGE NOTATION

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

def to_label_track(string):
	pattern = r'\s*(\S+)\s+(\S+)\s*(.*)'
	result = [re.match(pattern, x) for x in string.split('\n')]
	result = [list(x.groups()) for x in result if x is not None]
	result = [[float(x[0]), float(x[1])] + x[2:] for x in result]
	for i in range(len(result)):
		#print(i)
		if result[i][0] == result[i][1]:
			if i < len(result) - 1:
				result[i][1] = result[i+1][0]
			else:
				result = result[:len(result) - 1]
	return result

def camel_to_snake(name: str) -> str:
	# Insert underscore before capital letters, except the first character
	s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
	snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
	return snake

def snake_to_camel(name: str, lower=False) -> str:
	result = ''.join(word.capitalize() if word else '_' for word in name.split('_'))
	return result[0].lower() + result[1:] if lower else result

