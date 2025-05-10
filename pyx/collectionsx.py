import copy
import random
import numpy as np

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

class bag(list):	#each item in bag is a tuple like (item, count)
	def to_list(self):
		result = []
		for x in self:
			result += [x[0]] * x[1]
		return result

	def to_set(self): #return the underlying set
		return [x[0] for x in self]

def For(start, stop, step=None, func=None):
	if func is None:
		raise ValueError("You must provide a function")

	if step is None:
		step = np.ones_like(start)
	else:
		step = np.array(step)

	def recursive_loop(pos):
		if len(pos) == len(start):
			func(np.array(pos))
			return
		i = len(pos)
		val = start[i]
		while val < stop[i]:
			recursive_loop(pos + [val])
			val += step[i]

	recursive_loop([])

def Map(start, stop=None, step=None, func=None):
	start = np.array(start, dtype=int)
	if stop is None:
		stop = start
		start = np.zeros_like(stop)
	else:
		stop = np.array(stop, dtype=int)

	if step is None:
		step = np.ones_like(start, dtype=int)
	else:
		step = np.array(step, dtype=int)

	shape = ((stop - start + step - 1) // step).astype(int)
	result = np.empty(shape, dtype=object)

	def recurse(index, depth, out_view):
		if depth == len(start):
			out_view[...] = func(np.array(index))
			return
		i = start[depth]
		s = 0
		while i < stop[depth]:
			recurse(index + [i], depth + 1, out_view[s])
			i += step[depth]
			s += 1

	recurse([], 0, result)
	return result
