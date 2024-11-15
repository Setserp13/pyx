import math
from multipledispatch import dispatch
from numbers import Number
import numpy as np
from pyx.mat.mat import polar_to_cartesian
import itertools
from pyx.collectionsx import List as ls

def clamp(point, min, max): return np.minimum(np.maximum(point, min), max)

def lerp(a, b, t): return a * (1 - t) + b * t


def on_circle(n, r=1.0, center=np.zeros(2), start=0.0):
	return [np.array(polar_to_cartesian(r, start + 2.0 * math.pi * (i / n))) + center for i in range(n)]

def on_arc(n, r=1.0, center=np.zeros(2), start=0.0, size=2.0 * math.pi):
	return [np.array(polar_to_cartesian(r, start + size * (i / (n - 1)))) + center for i in range(n)]

class rect:
	def __init__(self, min, size):
		self.min = np.array(min)
		self.size = np.array(size)

	@property
	def center(self): return self.min + self.extents

	@property
	def extents(self): return self.size * 0.5

	@property
	def max(self): return self.min + self.size

	def normalize_point(self, value): return (value - self.min) / self.size

	def denormalize_point(self, value): return value * self.size + self.min

	def normalize_vector(self, value): return value / self.size

	def denormalize_vector(self, value): return value * self.size

	def normalize_rect(self, value):
		return rect(self.normalize_point(value.min), self.normalize_vector(value.size))

	def denormalize_rect(self, value):
		return rect(self.denormalize_point(value.min), self.denormalize_vector(value.size))

	def set_position(self, pivot, value): #pivot is normalized and value is not normalized
		return rect(self.min + (value - self.denormalize_point(pivot)), self.size)

	def clamp(self, point): return clamp(point, self.min, self.max)

	def contains_point(self, point):
		return contains(self.min, self.max, point)
		#return all(self.min[i] <= x and x <= self.max[i] for i, x in enumerate(point))

	def contains_rect(self, rect):
		return self.contains_point(rect.min) and self.contains_point(rect.max)

	#STATIC
	def center_size(center, size): return rect(center - size * 0.5, size)

	def min_max(min, max): return rect(min, max - min)

	def aabb(a, b):
		return rect.min_max(np.minimum(a.min, b.min), np.maximum(a.max, b.max))

	def distances(a, b):
		return [max(a.min[i] - b.max[i], b.min[i] - a.max[i], 0) for i in range(len(a.min))]

	def chebyshev_distance(a, b): return max(rect.distances(a, b))
	
	def euclidean_distance(a, b): return sqrt(sum(x ** 2 for x in rect.distances(a, b)))

	def manhattan_distance(a, b): return sum(rect.distances(a, b))

	def padding(self, left, right, relative=False):
		if relative:
			left = self.denormalize_vector(left)
			right = self.denormalize_vector(right)
		return rect.min_max(self.min + left, self.max - right)

	def expand(self, amount):
		padding = -np.full(len(self.min), amount)
		return self.padding(padding, padding)

	def bounds(self, obj_size, obj_pivot=None):
		if obj_pivot == None:
			obj_pivot = np.full(len(obj_size), 0.5)
		return self.padding(obj_size * obj_pivot, obj_size * (1 - obj_pivot))

	def axis_intersection(a, b, axis=0):
		start = max(a.min[axis], b.min[axis])
		stop = min(a.max[axis], b.max[axis])
		return None if stop < start else (start, stop)

	def intersection(a, b):
		start = []
		stop = []
		for i in range(len(a.min)):
			interval = rect.axis_intersection(a, b, i)
			if interval is None:
				return None
			start.append(interval[0])
			stop.append(interval[1])
		return rect.min_max(np.array(start), np.array(stop))

	def asint(self): return rect(self.min.astype(int), self.size.astype(int))

	def volume(self): return np.prod(self.size)

	def contains_rect_percent(self, other, percent=1.0):
		inter = rect.intersection(self, other)
		return False if inter is None else inter.volume() / other.volume() >= percent

	def subrects(self, ls, normalized=False):
		return [self.normalize_rect(x) if normalized else x for x in ls if self.contains_rect(x)]

	def slice_by_cell_count(self, cell_count): return self.slice_by_cell_size(self.size / cell_count)

	def slice_by_cell_size(self, cell_size):
		cell_count = np.ceil(self.size / cell_size).astype(int)
		"""result = []
		for x in itertools.product(*[range(x) for x in cell_count]):
			cell_min = self.min + np.array(x) * cell_size
			cell_max = self.min + (np.array(x) + np.ones(len(x))) * cell_size
			cell_max = np.minimum(cell_max, self.max)
			result.append(rect.min_max(cell_min, cell_max))"""
		return [ rect.min_max(x.min, np.minimum(x.max, self.max)) for x in grid(cell_size, offset=self.min).cells(cell_count)]
		return result

	def __repr__(self): return f'rect(min={self.min}, size={self.size}, max={self.max})'

	def __str__(self): return f'Min: {self.min}, Size: {self.size}, Max: {self.max}'



def rect2(x, y, width, height): return rect(np.array([x, y]), np.array([width, height]))

def rect3(x, y, z, width, height, depth): return rect(np.array([x, y, z]), np.array([width, height, depth]))

def bottom_left(rect): return rect.denormalize_point(np.array([0, 0]))
def bottom_right(rect): return rect.denormalize_point(np.array([0, 1]))
def top_left(rect): return rect.denormalize_point(np.array([1, 0]))
def top_right(rect): return rect.denormalize_point(np.array([1, 1]))
def corners(rect): return [bottom_left(rect), top_left(rect), top_right(rect), bottom_right(rect)]
def area(rect): return rect.size[0] * rect.size[1]

class grid:
	def __init__(self, cell_size, offset=None, cell_gap=None):
		self.cell_size = np.array(cell_size)
		self.offset = np.zeros(len(cell_size)) if offset is None else np.array(offset)
		self.cell_gap = np.zeros(len(cell_size)) if cell_gap is None else np.array(cell_gap)

	def cell_min(self, index):
		return self.offset + (self.cell_size + self.cell_gap) * index

	def cell_index(self, point):
		return (point - self.offset) // (self.cell_size + self.cell_gap)
	
	def cell(self, index):
		return rect(self.cell_min(index), self.cell_size)

	def lines(self, min_index, max_index, axis=None):
		result = []
		if axis is None:
			for i in range(len(min_index)):
				result += self.lines(min_index, max_index, i)
		else:
			for i in range(min_index[axis], max_index[axis]):
				a = np.array(min_index)
				b = np.array(max_index)
				a[axis] = b[axis] = i
				result.append([self.cell_min(a), self.cell_min(b)])
		return result

	def cells(self, stop, start=np.zeros(2), swizzle=[0,1]):
		start = ls.items(start, swizzle)
		stop = ls.items(stop, swizzle)
		ranges = [list(range(int(start[i]), int(stop[i]))) for i in range(len(start))]
		indices = itertools.product(*ranges)
		indices = [ls.items(x, swizzle) for x in indices]
		return [self.cell(np.array(x)) for x in indices]



#POINT-POINT AABB
@dispatch(np.ndarray, np.ndarray)
def aabb(a, b): return rect.min_max(np.minimum(a, b), np.maximum(a, b))

#RECT-POINT AABB
@dispatch(rect, np.ndarray)
def aabb(a, b): return rect.min_max(np.minimum(a.min, b), np.maximum(a.max, b))

@dispatch(list)
def aabb(*points):
	return rect.min_max(np.minimum.reduce(points), np.maximum.reduce(points))

from typing import Union
#@dispatch(Number, Number, Number)

def contains(min, max, value) -> bool:
	if all(isinstance(x, Number) for x in [min, max, value]):
		return min <= value <= max
	#else is instance of Union[np.ndarray, list, tuple]
	if len(min) != len(max) or len(min) != len(value):
		raise ValueError("min, max, and value must have the same length")
	return all(min[i] <= x <= max[i] for i, x in enumerate(value))

def normalize(x):
	magnitude = np.linalg.norm(x)
	return x if magnitude == 0 else x / magnitude

def angle(a, b):
	result = np.dot(a, b)
	result /= np.linalg.norm(a) * np.linalg.norm(b)
	result = np.clip(result, -1.0, 1.0)		# Clip the cosine value to the range [-1, 1] to avoid numerical errors
	return np.arccos(result)

def angle2(a, b):
	dot_product = np.dot(a, b)
	cross_product = np.cross(a, b)
	angle = np.arctan2(cross_product, dot_product)[2]  # Gives the angle in radians
	# Normalize the angle to [0, 2*pi]
	#print(angle)
	if angle < 0:
		angle += 2 * np.pi
	return angle

def rotate(v, angle_rad):
	rotation = np.exp(1j * angle_rad) # Create a complex number representing the rotation
	result = rotation * complex(v[0], v[1])
	return np.array([result.real, result.imag])

class bbox:
	def circle(cx, cy, r): return rect2(cx - r, cy - r, r * 2, r * 2)
	def ellipse(cx, cy, rx, ry): return rect2(cx - rx, cy - ry, rx * 2, ry * 2)




#SIZING

def set_size_component(size, index, value, preserve_aspect=True):
	result = np.copy(size)
	result[index] = value
	if preserve_aspect:
		scale = value / size[index]
		result[:index] *= scale
		result[index+1:] *= scale
	return result

def aspect(size): return size[1] / size[0]	#Aspect ratio

#def set_aspect(size, value): return np.array([size[0], int(size[0] * value)])

def fit_scale(viewport, viewbox, scale_method='meet'):	#scale_method in ['meet', 'slice']
	scale = viewport / viewbox
	return {'meet': min, 'slice': max}[scale_method](scale)	

def fit(viewport, viewbox, scale_method='meet'): return viewbox * fit_scale(viewport, viewbox, scale_method)
