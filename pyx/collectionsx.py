import copy
import random
import numpy as np
from pyx.array_utility import find_index

def pad(array, pad_width, fill): return array + [fill] * (pad_width - len(array))

class Dict:
	def items(dct, keys): return { x: dct[x] for x in keys if x in dct } #also works with list and tuple
	def get(dct, key, dflt_value=None): return dct[key] if key in dct else dflt_value

class List:
	def items(arr, indices): return [arr[x] for x in indices if x in range(len(arr))] #also works with dict
	def get(arr, index, dflt_value=None): return arr[index] if index in range(len(arr)) else dflt_value

	#Return a k-cycle
	def arange(ls, k, start=0): return [ls[(start + i) % len(ls)] for i in range(k)]

	def aranges(ls, k, start=0, cycle=True): return [List.arange(ls, k, start + i) for i in range(len(ls) - (0 if cycle else (k - 1)))]

	def batch(lst, size):
		for i in range(0, len(lst), size):
			yield lst[i:i+size]

def get_random(ls, amount):	#amount can be an int in [0, len(ls) - 1] or a float in [0.0, 1.0]
	count = int(len(ls) * amount) if isinstance(amount, float) else amount
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

	def append(self, item, count=1):
		index = find_index(self, lambda x: x[0] == item)
		if index > -1:
			self[index][1] += count
		else:
			super().append([item, count])

	def from_list(ls):
		result = bag()
		for x in ls:
			result.append(x)
		return result
	
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

"""def flatten(lst):
	result = []
	for item in lst:
		if isinstance(item, list):
			result.extend(flatten(item))
		else:
			result.append(item)
	return result"""

from collections.abc import Iterable
def flatten(ls, times=1):
	if times < 1:
		return ls
	result = []
	for item in ls:
		if isinstance(item, Iterable):
			result.extend(flatten(item, times-1))
		else:
			result.append(item)
	return result


class graph(list):
	def __init__(self, *args):
		super().__init__(args)
		self.adjacency = [[] for _ in range(len(args))]

	def append(self, item):
		super().append(item)
		self.adjacency.append([])

	def extend(self, items):
		for x in items:
			self.append(x)

	def insert(self, index, item):
		super().insert(index, item)
		self.adjacency.insert(index, [])
		for adj in self.adjacency:	#Shift up indices greater than inserted
			adj[:] = [j if j < index else j + 1 for j in adj]

	def remove_at(self, index):
		super().pop(index)
		self.adjacency.pop(index)
		for adj in self.adjacency:	#Clean all references to this index, shift down indices greater than removed
			adj[:] = [j if j < index else j - 1 for j in adj if j != index]

	def remove(self, item):
		if item in self:
			index = self.index(item)
			self.remove_at(index)

	def add_arrow(self, i, j): self.adjacency[i].append(j)

	def add_edge(self, i, j):
		self.add_arrow(i, j)
		self.add_arrow(j, i)

	def add_arrows(self, args):
		for x in args: self.add_arrow(*x)

	def add_edges(self, args):
		for x in args: self.add_edge(*x)

	def neighbors_at(self, i): return List.items(self, self.adjacency[i])

	def neighbors(self, item):
		if item in self:
			return self.neighbors_at(self.index(item))

	def spanning_tree(self, start=0):
		n = len(self.adjacency)
		visited = [False]*n
		tree_edges = []
		def dfs(u):
			visited[u] = True
			for v in self.adjacency[u]:
				if not visited[v]:
					tree_edges.append((u, v))
					dfs(v)
		dfs(start)
		#return tree_edges
		result = graph(*self)
		result.add_edges(tree_edges)
		return result

	def arrows_from(self, i): return [(i, j) for j in self.adjacency[i]]
	#def edges_at(self, i): return [{i, j} for j in self.adjacency[i]]
	def arrows(self): return flatten([self.arrows_from(i) for i in range(len(self))])
	def edges(self): return list({tuple(sorted(e)) for e in self.arrows()})
	def get_edges(self): return [[self[i], self[j]] for i, j in self.edges()]

def list_equal(a, b, equal=np.array_equal):
	return all(equal(a[i], b[i]) for i in range(len(a))) if len(a) == len(b) else False

def contains(a, b, equal=np.array_equal): return all(any(equal(x, y) for y in a) for x in b)

def set_equal(a, b, equal=np.array_equal):
	return contains(a, b, equal=equal) and contains(b, a, equal=equal)

def distinct(ls, equal=np.array_equal):
	result = []
	for x in ls:
		#print(x)
		if not any(equal(x, y) for y in result):
			result.append(x)
	return result

def merge_where(ls, condition, merge_func):
	for i in range(len(ls) - 1, 0, -1):
		for j in range(i - 1, -1, -1):
			if condition(ls[i], ls[j]):
				ls[j] = merge_func(ls[i], ls[j])
				del ls[i]	#ls.pop(i)
				break
	return ls

"""
List (a.k.a. Sequence) → Order matters, Duplicates matter.
Bag (a.k.a. Multiset) → Order does not matter, Duplicates matter.
Set → Order does not matter, Duplicates do not matter.
"""

