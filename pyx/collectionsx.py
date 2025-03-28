import copy
import random

class Dict:
	def items(dct, keys): return { x: dct[x] for x in keys if x in dct } #also works with list and tuple
	def get(dct, key, dflt_value=None): return dct[key] if key in dct else dflt_value

class List:
	def items(arr, indices): return [arr[x] for x in indices if x in range(len(arr))] #also works with dict
	def get(arr, index, dflt_value=None): return arr[index] if index in range(len(arr)) else dflt_value
	

def get_random(ls, count):
	result = copy.copy(ls)
	for i in range(len(ls) - count):
		result.pop(random.randint(0, len(result)-1))
	return result

def lshift(arr, n=1): return arr[n:] + arr[:n]		#Left shift the array arr by n positions

def rshift(arr, n=1): return arr[-n:] + arr[:-n]	#Right shift the array arr by n positions
