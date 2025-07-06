import copy
import random
import numpy as np

class Dict:
	def items(dct, keys): return { x: dct[x] for x in keys if x in dct } #also works with list and tuple
	def get(dct, key, dflt_value=None): return dct[key] if key in dct else dflt_value

class List:
	def items(arr, indices): return [arr[x] for x in indices if x in range(len(arr))] #also works with dict
	def get(arr, index, dflt_value=None): return arr[index] if index in range(len(arr)) else dflt_value

	#Return a k-cycle
	def arange(ls, k, start=0): return [ls[(start + i) % len(ls)] for i in range(k)]

	def aranges(ls, k, cycle=True): return [List.arange(ls, k, i) for i in range(len(ls) - (0 if cycle else (k - 1)))]
	

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

from itertools import product

def Map(start, stop=None, step=None, func=None):
    # Converte para listas de inteiros
    start = list(map(int, start))
    
    if stop is None:
        stop = start
        start = [0] * len(stop)
    else:
        stop = list(map(int, stop))
    
    if step is None:
        step = [1] * len(start)
    else:
        step = list(map(int, step))
    
    # Calcula o shape
    shape = []
    for s, e, st in zip(start, stop, step):
        dim = (e - s + st - 1) // st
        shape.append(dim)

    # Inicializa a matriz com listas aninhadas
    def create_nested_list(shape, level=0):
        if level == len(shape) - 1:
            return [None] * shape[level]
        return [create_nested_list(shape, level + 1) for _ in range(shape[level])]

    result = create_nested_list(shape)

    # Função para acessar e modificar valor por índice múltiplo
    def set_value(container, idx, value):
        for i in idx[:-1]:
            container = container[i]
        container[idx[-1]] = value

    # Preenche o resultado com os valores
    for offset in product(*[range(s) for s in shape]):
        index = [start[i] + step[i] * offset[i] for i in range(len(start))]
        set_value(result, offset, func(*index))	#index))

    return result

def flatten(lst):
	result = []
	for item in lst:
		if isinstance(item, list):
			result.extend(flatten(item))
		else:
			result.append(item)
	return result
