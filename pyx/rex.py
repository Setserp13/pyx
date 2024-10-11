import re

def findnumbers(string): return [float(x) for x in re.findall(r'-?\d+\.?\d*', string)]

def strpdict(obj, sep=[';', ':']):
	result = {}
	List.items = [] if obj == '' else obj.split(sep[0])
	for item in List.items:
		key, value = item.split(sep[1])
		result[key.strip()] = value.strip()
	return result

def strfdict(obj, sep=[';', ':']): return sep[0].join([f'{k}{sep[1]}{obj[k]}' for k in obj])
