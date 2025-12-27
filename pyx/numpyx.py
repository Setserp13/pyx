import math
from multipledispatch import dispatch
from numbers import Number
import numpy as np
import itertools
from pyx.collectionsx import List as ls
import random
from pyx.numpyx_geo import polyline, line

def project(v, u): return (np.dot(v, u) / np.dot(u, u)) * u	#Project vector v onto vector u.

def fill(obj, length, filler=0):
	if isinstance(obj, np.ndarray):
		return np.append(obj, [filler] * (length - len(obj)))
	return [fill(x, length, filler) for x in obj]

#works for both scalars (numbers) and NumPy arrays
def clamp(value, min, max): return np.minimum(np.maximum(value, min), max)

def clamp01(value): return clamp(value, 0.0, 1.0)

def repeat(t, length):
	return clamp(t - math.floor(t / length) * length, 0.0, length)

def ping_pong(t, length):
	t = repeat(t, length * 2)
	return length - abs(t - length)

def clamp_magnitude(v, max_magnitude):
	current_magnitude = np.linalg.norm(v)
	if current_magnitude > max_magnitude:
		v = (v / current_magnitude) * max_magnitude
	return v

@dispatch(np.ndarray, np.ndarray, float)
def lerp(a, b, t): return a * (1 - t) + b * t	#works for both Number and ndarray

@dispatch(Number, Number, float)
def lerp(a, b, t): return a * (1 - t) + b * t	#a + (b - a) * t

@dispatch(int, int, float)
def lerp(a, b, t): return int(a * (1 - t) + b * t)

"""@dispatch(list, list, float)
def lerp(a, b, t): return list(map(lambda x, y: lerp(x, y, t), a, b))
@dispatch(tuple, tuple, float)
def lerp(a, b, t): return tuple(map(lambda x, y: lerp(x, y, t), a, b))"""

@dispatch(Number, Number, Number)
def inverse_lerp(a, b, c): return (c - a) / (b - a)

@dispatch(np.ndarray, np.ndarray, np.ndarray)
def inverse_lerp(a, b, c):	#Works even if c is not exactly on the line (you get the fractional position along the segment).
	ab = b - a
	t = np.dot(c - a, ab) / np.dot(ab, ab)
	return t
	#return distance(a, c) / distance(a, b)

def raised_norm(a, n=2): return sum(abs(x) ** n for x in a)	#default n=2 means sqr magnitude
def norm(a, n=2): return raised_norm(a, n) ** (1.0 / n)	#default n=2 means magnitude
def raised_distance(a, b, n=2): return raised_norm(b - a, n)	#default n=2 means sqr euclidean distance
def distance(a, b, n=2): return raised_distance(a, b, n) ** (1.0 / n)	#default n=2 means euclidean distance

def subdivide(a, b, n):
	if n == 1:
		return [a]
	return [lerp(a, b, i / (n - 1)) for i in range(n)]

def linear_layout(n, offset=np.zeros(3), dir=np.array([1, 0, 0]), cell_size=1, align=0.5):
	dir = normalize(dir)
	size = cell_size * (n - 1)
	offset = offset - size * align * dir
	return subdivide(offset, offset + size * dir, n)

@dispatch(np.ndarray, np.ndarray)
def random_range(start, stop): return np.array(list(map(lambda x, y: random_range(x, y), start, stop)))

def on_circle(n, r=1.0, center=np.zeros(2), start=0.0):	#regular polygon
	return polyline([polar_to_cartesian(r, start + 2.0 * math.pi * (i / n)) + center for i in range(n)])

def on_arc(n, r=1.0, center=np.zeros(2), start=0.0, size=2.0 * math.pi):
	return polyline([polar_to_cartesian(r, start + size * (i / (n - 1))) + center for i in range(n)])

def on_sphere(radius=1.0, stacks=16, slices=32, center=np.zeros(3)):
	v = []
	for theta in subdivide(0.0, math.pi, stacks):
		for phi in subdivide(0.0, math.pi * 2.0, slices):
			v.append(spherical_to_cartesian(radius, theta, phi) + center)
	return v

class rect:
	def __init__(self, min, size):
		self.min = np.array(min)
		self.size = np.array(size)

	@property
	def center(self): return self.min + self.extents
	@center.setter
	def center(self, value): self.min = value - self.extents
	
	@property
	def extents(self): return self.size * 0.5
	@extents.setter
	def extents(self, value): self.size = value * 2
	
	@property
	def max(self): return self.min + self.size
	@max.setter
	def max(self, value): self.min = value - self.size

	def normalize_point(self, value): return (value - self.min) / self.size
	def normalize_point_component(self, value, axis=0): return (value - self.min[axis]) / self.size[axis]

	def denormalize_point(self, value): return value * self.size + self.min
	def denormalize_point_component(self, value, axis=0): return value * self.size[axis] + self.min[axis]

	def normalize_vector(self, value): return value / self.size

	def denormalize_vector(self, value): return value * self.size

	def normalize_rect(self, value):
		return rect(self.normalize_point(value.min), self.normalize_vector(value.size))

	def denormalize_rect(self, value):
		return rect(self.denormalize_point(value.min), self.denormalize_vector(value.size))

	def set_axis_position(self, pivot, value, axis=0): #pivot is normalized and value is not normalized
		result = rect(self.min, self.size)
		result.min[axis] += (value - result.denormalize_point_component(pivot, axis=axis))
		return result
		
	def set_position(self, pivot, value): #pivot is normalized and value is not normalized
		return rect(self.min + (value - self.denormalize_point(pivot)), self.size)

	def clamp(self, point): return clamp(point, self.min, self.max)

	def clamp_rect(self, value): return rect.min_max(self.clamp(value.min), self.clamp(value.max))
	
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
		if obj_pivot is None:
			obj_pivot = np.full(len(obj_size), 0.5)
		return self.padding(obj_size * obj_pivot, obj_size * (1 - obj_pivot))

	def random_point(self): return random_range(self.min, self.max)
		
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

	def slice_by_cell_count(self, cell_count): return self.slice_by_cell_size(self.size / np.array(cell_count, dtype=float))

	def slice_by_cell_size(self, cell_size):
		cell_count = np.ceil(self.size / cell_size).astype(int)
		return [ rect.min_max(x.min, np.minimum(x.max, self.max)) for x in grid(cell_size, offset=self.min).cells(cell_count)]
		return result

	def __repr__(self): return f'rect(min={self.min}, size={self.size}, max={self.max})'

	def __str__(self): return f'Min: {self.min}, Size: {self.size}, Max: {self.max}'

	def __add__(self, vector): return rect(self.min + vector, self.size)

	def __sub__(self, vector): return rect(self.min - vector, self.size)

	def axis_scale(self, value, pivot=0.5, axis=0):
		delta = value - self.size[axis]
		self.min[axis] -= delta * pivot
		self.size[axis] = value
	
	def scale(self, value, pivot=np.full(2, 0.5)):
		delta = value - self.size
		self.min -= delta * pivot
		self.size = value

	def face_center(rect, axis, dir):  # dir -> -1: left/down/back and so on..., 1: right/up/front and so on...
		return rect.center + rect.extents[axis] * ei(axis, len(rect.center)) * dir
	
	def face(rect, axis, dir):  # dir -> -1: left/down/back, 1: right/up/front
		size = np.concatenate((rect.size[:axis], [0.0], rect.size[axis+1:]), axis=0)
		return rect.center_size(rect.face_center(axis, dir), size)



class rect2(rect):
	def __init__(self, x, y, width, height):
		super().__init__(np.array([x, y]), np.array([width, height]))

	"""def bottom_left(rect): return rect.denormalize_point(np.array([0, 0]))
	def bottom_right(rect): return rect.denormalize_point(np.array([0, 1]))
	def top_left(rect): return rect.denormalize_point(np.array([1, 0]))
	def top_right(rect): return rect.denormalize_point(np.array([1, 1]))
	def left(rect): return [rect2.bottom_left(rect), rect2.top_left(rect)]
	def right(rect): return [rect2.bottom_right(rect), rect2.top_right(rect)]
	def bottom(rect): return [rect2.bottom_left(rect), rect2.bottom_right(rect)]
	def top(rect): return [rect2.top_left(rect), rect2.top_right(rect)]"""
	def bottom_left(rect): return rect.denormalize_point(np.array([0, 0]))
	def bottom_right(rect): return rect.denormalize_point(np.array([1, 0]))
	def top_left(rect): return rect.denormalize_point(np.array([0, 1]))
	def top_right(rect): return rect.denormalize_point(np.array([1, 1]))
	
	def side(rect, axis, dir):  # dir -> 0: left/down, 1: right/up
		return line([
			rect.denormalize_point(np.array([dir, i])[[axis, 1 - axis]])
			for i in [0.0, 1.0]
		])
	def left(rect): return rect2.side(rect, axis=0, dir=0)
	def right(rect): return rect2.side(rect, axis=0, dir=1)
	def bottom(rect): return rect2.side(rect, axis=1, dir=0)
	def top(rect): return rect2.side(rect, axis=1, dir=1)

	def corners(rect): return polyline([rect2.bottom_left(rect), rect2.top_left(rect), rect2.top_right(rect), rect2.bottom_right(rect)])
	def area(rect): return rect.size[0] * rect.size[1]
	def cut(rect, t, axis=0, expand=0):
		u = ei(axis, 2) * t
		v = ei(1 - axis, 2)
		return [rect.denormalize_point(u + v * 0) - v * expand, rect.denormalize_point(u + v) + v * expand]

def rect3(x, y, z, width, height, depth): return rect(np.array([x, y, z]), np.array([width, height, depth]))

class grid:
	def __init__(self, cell_size, offset=None, cell_gap=None):
		self.cell_size = np.array(cell_size)
		self.offset = np.zeros(len(cell_size)) if offset is None else np.array(offset)
		self.cell_gap = np.zeros(len(cell_size)) if cell_gap is None else np.array(cell_gap)

	def cell_min(self, index):
		return self.offset + (self.cell_size + self.cell_gap) * index

	def cell_index(self, point):
		return (point - self.offset) // (self.cell_size + self.cell_gap)

	def snap_point(self, point, to=np.zeros(2)):
		index = self.cell_index(point).astype(int)
		return self.cell(index).denormalize_point(to)

	def snap_points(self, points, to=np.zeros(2)): return [self.snap_point(x, to) for x in points]

	def snap_rect(self, rct): return rect.min_max(self.snap_point(rct.min, np.zeros(2)), self.snap_point(rct.max, np.ones(2)))
		
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

#RECT-RECT AABB
@dispatch(rect, rect)
def aabb(a, b): return rect.min_max(np.minimum(a.min, b.min), np.maximum(a.max, b.max))

#@dispatch(list)
#def aabb(*points):
#	return rect.min_max(np.minimum.reduce(points), np.maximum.reduce(points))

@dispatch(list)
def aabb(args): #args can contain np.ndarray and rect
	if len(args) == 1: return aabb(args[0], args[0])
	result = aabb(args[0], args[1])
	for i in range(2, len(args)):
		result = aabb(result, args[i])
	return result

@dispatch(np.ndarray)
def aabb(arr): #args can contain np.ndarray and rect
	if arr.ndim != 2:
		raise TypeError("Expected 2D array")
	return aabb(list(arr))

def set_aabb(p, value):	#p is a list of points
	current = aabb(p)
	return [value.denormalize_point(current.normalize_point(x)) for x in p]

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
	angle = np.arctan2(cross_product, dot_product)  # Gives the angle in radians
	# Normalize the angle to [0, 2*pi]
	#print(angle)
	if angle < 0:
		angle += 2 * np.pi
	return angle

def rotate(v, angle_rad):
	rotation = np.exp(1j * angle_rad) # Create a complex number representing the rotation
	result = rotation * complex(v[0], v[1])
	return np.array([result.real, result.imag])

def rotate_around(point, angle, center=np.zeros(2)): #use default center to rotate vectors
	return rotate(point - center, angle) + center



def align(object, anchor, pivot, value): return object.set_position(pivot, anchor.denormalize_point(value))

def align_component(object, anchor, pivot, value, axis=0): return object.set_axis_position(pivot, anchor.denormalize_point_component(value, axis=axis), axis=axis)

#standard unit vector, where dim is dimension
def ei(axis=0, dim=3): return np.array([1.0 if i == axis else 0.0 for i in range(dim)])

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

def fit_rect(viewport_rect, viewbox, mode='slice', align=0.5):
	result = rect(viewport_rect.min, fit(viewport_rect.size, viewbox, mode))
	sx = viewbox[0] / viewport_rect.size[0]	# Scale factors
	sy = viewbox[1] / viewport_rect.size[1]
	# Axis that overflows and needs alignment
	axis = 0 if (mode == 'slice' and sy < sx) or (mode == 'meet' and sy > sx) else 1
	# available slack along that axis
	slack = viewport_rect.size[axis] - result.size[axis]
	# shift according to align (0 = min aligned, 1 = max aligned, .5 = centered)
	result.min[axis] = viewport_rect.min[axis] + slack * align
	return result



def rotate_2d(v, theta): #theta is in radians
	x, y = v
	c, s = math.cos(theta), math.sin(theta)
	return np.array([x * c - y * s, x * s + y * c])


def random_on_arc(r, start_angle, stop_angle): return polar_to_cartesian(r, random.uniform(start_angle, stop_angle))
def random_in_arc(r, start_angle, stop_angle): return random_on_arc(random.uniform(0, r), start_angle, stop_angle)
def random_on_circle(r): return random_on_arc(r, 0, 2 * math.pi)
def random_in_annulus(r, R): return random_on_circle(random.uniform(r, R))
def random_in_circle(r): return random_in_annulus(0, r)
def uniform_in_annulus(r, R): return random_on_circle(lerp(r, R, math.sqrt(random.random())))
def uniform_in_circle(r): return uniform_in_annulus(0, r)
def random_in_segment(a, b): return lerp(a, b, random.uniform(0.0, 1.0))

def polar_to_cartesian(r, theta): return np.array([r * math.cos(theta), r * math.sin(theta)])
def cartesian_to_polar(x, y): return np.array([math.sqrt(x**2 + y**2), math.atan2(y, x)])


def cartesian_to_spherical(x, y, z):
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arccos(z / r) if r != 0 else 0  # Inclinação
	phi = np.arctan2(y, x)  # Azimute
	return r, theta, phi

def spherical_to_cartesian(r, theta, phi):	#radius r, inclination θ, azimuth φ, where r ∈ [0, ∞), θ ∈ [0, π], φ ∈ [0, 2π)
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	return np.array([x, y, z])

def random_on_sphere(r): return spherical_to_cartesian(r, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))
def random_in_sphere(r): return random_on_sphere(random.uniform(0, r))

def random_on_line(start, end): return lerp(start, end, random.uniform(0.0, 1.0))



def find_indices(array, match, max_dim=None):
	shape = array.shape if max_dim is None else array.shape[:max_dim]
	result = []
	for idx in itertools.product(*(range(s) for s in shape)):
		# Expand idx with full slices for remaining dimensions
		slicing = idx + (Ellipsis,)
		value = array[slicing]
		if match(value):
			result.append(np.array(idx))
	return result

def row_major_order(size, index):	#linear = i0*(s1*s2*...*sn) + i1*(s2*s3*...*sn) + ... + in
    """
    Compute linear index of an N-dimensional index (row-major order)

    size  – ndarray of shape (N,) with sizes of each dimension
    index – ndarray of shape (N,) with indices in each dimension
    """
    size = np.asarray(size)
    index = np.asarray(index)
    multipliers = np.cumprod(size[::-1])[::-1][1:]
    multipliers = np.append(multipliers, 1)  # last multiplier = 1
    return int(np.sum(index * multipliers))

def column_major_order(size, index):	#linear = i0 + i1*s0 + i2*(s0*s1) + ...
    """
    Compute linear index of an N-dimensional index (column-major order)

    size  – ndarray of shape (N,)
    index – ndarray of shape (N,)
    """
    size = np.asarray(size)
    index = np.asarray(index)
    multipliers = np.cumprod(size) / size  # produces: [1, s0, s0*s1, ...]
    multipliers = multipliers.astype(int)
    return int(np.sum(index * multipliers))
