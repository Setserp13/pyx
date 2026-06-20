import numpy as np
import pyx.numpyx as npx
import pyx.mat.mat as mat
from pyx.array_utility import find_indices
from pyx.collectionsx import flatten, List, lshift, graph
import math
import pyx.osx as osx
import itertools
from itertools import product
from multipledispatch import dispatch
import copy
from pyx.mat.transform import Transform
import pyx.PILx as PILx

EPSILON = 1e-9
TAU = math.pi * 2.

class shape:
    def __add__(self, value):
        raise NotImplementedError

    def __sub__(self, value):
        raise NotImplementedError

    def __mul__(self, value):
        raise NotImplementedError

    __rmul__ = __mul__

    def scale(self, factor, pivot=None):
        if pivot is None:
            pivot = np.zeros(self.ndim)

        result = copy.deepcopy(self)
        return pivot + (result - pivot) * factor


class rect_like:
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
	def extents(self, value): self.size = value * 2.
	
	@property
	def max(self): return self.min + self.size
	@max.setter
	def max(self, value): self.min = value - self.size

	@classmethod
	def min_size(cls, min, size):
		obj = cls.__new__(cls)     # cria objeto da classe correta
		rect_like.__init__(obj, min, size)  # inicializa como rect_like
		return obj

	@classmethod
	def min_max(cls, min, max): return cls.min_size(min, max - min)

	@classmethod
	def center_size(cls, center, size): return cls.min_size(center - size * 0.5, size)

	@property
	def aabb(self): return rect(self.min, self.size)
	@aabb.setter
	def aabb(self, value):
		self.min = value.min
		self.size = value.size

	def __add__(self, vector): return type(self).min_size(self.min + vector, self.size)

	def __sub__(self, vector): return self.__add__(-vector)
	
	@property
	def ndim(self): return len(self.min)

	def __rmatmul__(self, M): return self.__matmul__(M)

	def __matmul__(self, M):	#Applies a (n+1)x(n+1) affine transform
		M = np.asarray(M, dtype=float)
		if M.shape != (self.ndim + 1, self.ndim + 1):
			raise ValueError("Expected a (n+1)x(n+1) affine matrix")

		# position (affected by translation)
		min_h = np.append(self.min, 1)
		min = (M @ min_h)[:self.ndim]

		# size (direction vector, no translation)
		size_h = np.append(self.size, 0)
		size = (M @ size_h)[:self.ndim]

		return type(self).min_size(min, size)
		
	def copy(self): return type(self).min_size(self.min.copy(), self.size.copy())

	def __repr__(self): return f'{type(self).__name__}(min={self.min}, size={self.size})'

	def __str__(self): return self.__repr__()

	def astype(self, dtype): return type(self).min_size(self.min.astype(dtype), self.size.astype(dtype))

	def asint(self): return self.astype(int)


class rect(rect_like):
	
	def normalize_point(self, value):
		size = self.size
		size = np.where(size == 0, 1, size)
		return (value - self.min) / size
	def normalize_point_component(self, value, axis=0): return (value - self.min[axis]) / self.size[axis]

	def denormalize_point(self, value): return value * self.size + self.min
	def denormalize_point_component(self, value, axis=0): return value * self.size[axis] + self.min[axis]

	def denormalize_points(self, value): return [self.denormalize_point(x) for x in value]
	
	def normalize_vector(self, value):
		size = self.size
		size = np.where(size == 0, 1, size)
		return value / size

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

	def random_point(self): return np.random.uniform(self.min, self.max)	#npx.random_range(self.min, self.max)
		
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

	def ndface_centers(self, axes):	#The faces are parallel to the Euclidean space defined by the axes
		p = [[0.5] if i in axes else [0.0, 1.0] for i in range(self.ndim)]
		#print(p)
		return [self.denormalize_point(np.array(x)) for x in product(*p)]

	def ndfaces(self, axes):
		centers = self.ndface_centers(axes)
		return [rect.center_size(center, np.array([self.size[i] if i in axes else 0.0 for i in range(self.ndim)])) for center in centers]

	@property
	def vertices(self): return points(product(*zip(self.min, self.max)))
	
	@property
	def local_aabb(self): return self.vertices.local_aabb

	@property
	def global_aabb(self): return self.vertices.global_aabb



class rect2(rect):
	def __init__(self, x, y, width, height):
		super().__init__(np.array([x, y]), np.array([width, height]))

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

	def corners(rect): return polyline([rect2.bottom_left(rect), rect2.top_left(rect), rect2.top_right(rect), rect2.bottom_right(rect)], closed=True)
	def area(rect): return rect.size[0] * rect.size[1]

	def line_at(rect, t, axis=0):	#result is a line parallel to axis
		u = ei(1 - axis, 2) * t
		v = ei(axis, 2)
		return line([rect.denormalize_point(u + v * i) for i in range(2)])

	def lines(rect, n, axis=0):
		return [rect2.line_at(rect, t, axis=axis) for t in subdivide(0.0, 1.0, n)]

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
				result.append(line([self.cell_min(a), self.cell_min(b)]))
		return result

	def cells(self, stop, start=None, swizzle=None):
		start = np.zeros(len(stop)) if start is None else start
		swizzle = list(range(len(stop))) if swizzle is None else swizzle
		start = List.items(start, swizzle)
		stop = List.items(stop, swizzle)
		ranges = [list(range(int(start[i]), int(stop[i]))) for i in range(len(start))]
		indices = itertools.product(*ranges)
		indices = [List.items(x, swizzle) for x in indices]
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




class points(np.ndarray):

	def __new__(cls, input_array, **attrs):
		obj = np.asarray(input_array).view(cls)

		for k,v in attrs.items():
			setattr(obj, k, v)

		return obj

	def __array_finalize__(self, obj):

		if obj is None:
			return

		# copy attributes automatically
		for k, v in getattr(obj, "__dict__", {}).items():
			setattr(self, k, v)

	@property
	def aabb(self):
		return aabb(self)

	@aabb.setter
	def aabb(self, value):
		self[:] = set_aabb(self, value)

	@property
	def local_aabb(self):
		M = self.attrib.get("transform")
		return self.aabb if M is None else (self @ M.TRS).aabb

	@property
	def global_aabb(self):
		M = self.attrib.get("transform")
		return self.aabb if M is None else (self @ M.global_TRS).aabb

	def transform(self, M):
		pts = npx.affine_transform(M, self)
		return type(self)(pts)

	def __matmul__(self, M):
		return self.transform(M)

	def __rmatmul__(self, M):
		return self.transform(M)

	def copy(self):
		obj = type(self)(np.array(self))
		obj.__dict__.update(self.__dict__)
		return obj

	def __repr__(self):
		name = type(self).__name__
		return f"{name}({np.asarray(self)}, attrs={self.__dict__})"

	@property
	def vertex_centroid(p):
		return np.mean(p, axis=0)
	
	@vertex_centroid.setter
	def vertex_centroid(p, value):
		p[:] += value - p.vertex_centroid

def elbow_connector1(start, end, x=0): return polyline([start, np.array([start[x], end[1 - x]]), end])

def elbow_connector2(start, end, x=0, t=0.5):
	y = npx.lerp(start[1 - x], end[1 - x], t)
	return polyline([start, np.array([start[x], y]), np.array([end[x], y]), end])

#In geometry, a set of points are said to be concyclic (or cocyclic) if they lie on a common circle. A polygon whose vertices are concyclic is called a cyclic polygon, and the circle is called its circumscribing circle or circumcircle.
def cyclic_polygon(angles, r=1, center=np.zeros(2)):
	return [npx.polar_to_cartesian(r, x) + center for x in angles]

def radar_polygon(radii, center=np.zeros(2)):
	angles = np.arange(0.0, math.pi * 2.0, math.pi * 2.0 / len(radii))
	return [npx.polar_to_cartesian(r, theta) + center for r, theta in zip(radii, angles)]

class circle():
	def __init__(self, center, radius):
		self.center = np.array(center)
		self.radius = radius

	def angle_of(self, point):
		delta = point - self.center
		return np.arctan2(delta[1], delta[0])

	def get_point(self, theta): return npx.polar_to_cartesian(self.radius, theta) + self.center		

	def get_point01(self, t): return self.get_point(t * 2 * np.pi)

	@property
	def diameter(self): return self.radius * 2

	@property
	def perimeter(self): return self.diameter * math.pi

	@property
	def area(self): return 2 * math.pi * self.radius ** 2

	def tangent(self, theta):
		p = self.get_point(theta)
		r = p - self.center
		t = np.array([-r[1], r[0]])  # 90° rotation
		return line([p, p + t])
	
	@property
	def aabb(self): return rect.center_size(self.center, np.ones(2) * self.diameter)
	@aabb.setter
	def aabb(self, value):
		self.center = value.center
		self.radius = min(*value.extents)

	def __rmatmul__(self, M): return self.__matmul__(M)

	def __matmul__(self, M):
		M = np.asarray(M, dtype=float)

		if M.shape != (3, 3):
			raise ValueError("Expected a 3x3 affine matrix")

		# Transform center (with translation)
		center_h = np.append(self.center, 1)
		center = (M @ center_h)[:2]

		# Extract linear part
		A = M[:2, :2]

		# Scale radius (assumes uniform scaling)
		scale_x = np.linalg.norm(A[:, 0])
		scale_y = np.linalg.norm(A[:, 1])

		if not np.isclose(scale_x, scale_y):
			raise ValueError("Non-uniform scaling turns a Circle into an Ellipse")

		radius *= scale_x
		return circle(center, radius)
		
	def copy(self): return circle(self.center.copy(), self.radius)

	def __add__(self, vector):
		obj = copy.deepcopy(self)
		obj.center += vector
		return obj

	def __sub__(self, vector): return self.__add__(-vector)


class ellipse(rect_like):

	def __init__(self, center, size):
		super().__init__(center - size * 0.5, size)

	@property
	def a(self): return max(self.extents)	# semi-major

	@property
	def b(self): return min(self.extents)	# semi-minor

	@property
	def major_axis(self):
		d = npx.ei(self.orientation, 2) * self.a
		return line(self.center + np.array([-d, d]))

	@property
	def minor_axis(self):
		d = npx.ei(1 - self.orientation, 2) * self.b
		return line(self.center + np.array([-d, d]))

	@property
	def orientation(self): return 0 if self.size[0] >= self.size[1] else 1

	@property
	def c(self): return np.sqrt(self.a**2 - self.b**2)	# Focal distance

	@property
	def foci(self):
		"""
		Returns both foci as np arrays
		"""
		d = npx.ei(self.orientation, 2) * self.c
		#print(d)
		return self.center + np.array([-d, d])

	@property
	def eccentricity(self): return self.c / self.a

	@property
	def area(self): return np.pi * self.a * self.b

	@property
	def perimeter(self):
		# Ramanujan approximation
		h = ((self.a - self.b)*2) / ((self.a + self.b)*2)
		return np.pi * (self.a + self.b) * (1 + (3*h)/(10 + np.sqrt(4 - 3*h)))

	def directrices(self):
		"""
		Returns the two directrix lines as (point, direction)
		"""
		e = self.eccentricity
		d = self.a / e
		ei = npx.ei(self.orientation, 2)
		return [line([self.center + ei * d * sgn, ei]) for sgn in [-1, 1]]

	# =========================
	# Parametric point
	# =========================
	def get_point(self, t):	#Standard parametric equation (centered)
		return np.array([np.cos(t), np.sin(t)]) * self.extents + self.center
	
	# =========================
	# Radius from focus (Kepler form)
	# =========================
	def radius_from_focus(self, theta):	#ellipse_radius
		"""
		Polar form relative to focus (important for orbits)
		"""
		e = self.eccentricity
		return self.a * (1 - e**2) / (1 + e * np.cos(theta))


	def point_from_focus_angle(self, theta, focus_index=0):
		"""
		Return point on ellipse given the angle from a focus (true anomaly)
		"""
	
		r = self.radius_from_focus(theta)
	
		focus = self.foci[focus_index]
	
		v = np.array([np.cos(theta), np.sin(theta)])
	
		# flip direction if using second focus
		if focus_index == 0:
			v = -v
	
		if self.orientation == 1:
			v = v[::-1]
	
		return focus + r * v

	
	@classmethod
	def from_a_e(cls, a, e, center=(0, 0), orientation=0):
		"""
		Create an ellipse from:
			a: semi-major axis
			e: eccentricity (0 <= e < 1)
			center: (cx, cy)
		horizontal: major axis orientation
		"""
		if not (0 <= e < 1):
			raise ValueError("Eccentricity must be in [0, 1)")

		b = a * (1 - e*2) * 0.5
		return cls(center=center, size=np.array([2 * a, 2 * b])[[orientation, 1 - orientation]])


	# =========================
	# True anomaly from point
	# =========================
	def true_anomaly(self, p, focus_index=0):
		"""
		Angle of a point on the ellipse measured from a focus
		"""
		f = self.foci[focus_index]
		v = np.array(p) - f
		return np.arctan2(v[1], v[0])
	
	
	# =========================
	# True anomaly -> eccentric anomaly
	# =========================
	def eccentric_anomaly(self, theta):
		e = self.eccentricity
		return 2 * np.arctan2(
			np.sqrt(1 - e) * np.sin(theta/2),
			np.sqrt(1 + e) * np.cos(theta/2)
		)
	
	# =========================
	# Eccentric anomaly -> true anomaly
	# =========================
	def true_from_eccentric(self, E):
		e = self.eccentricity
		return 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E/2))
	
	
	# =========================
	# Sector area from focus
	# =========================
	def sector_area_from_focus(self, theta1, theta2):
		a = self.a
		b = self.b
		e = self.eccentricity
	
		E1 = self.eccentric_anomaly(theta1)
		E2 = self.eccentric_anomaly(theta2)
	
		return (a*b/2) * ((E2 - e*np.sin(E2)) - (E1 - e*np.sin(E1)))
	
	
	# =========================
	# Area swept from focus starting at theta0
	# =========================
	def angles_from_area(self, area, theta0=0):
		"""
		Return (theta_start, theta_end) that sweep a given area
		from a focus.
		"""
	
		a = self.a
		b = self.b
		e = self.eccentricity
	
		E0 = self.eccentric_anomaly(theta0)
	
		target = area
	
		# initial guess
		E = E0 + target/(a*b/2)
	
		for _ in range(10):
			f = (a*b/2)*((E - e*np.sin(E)) - (E0 - e*np.sin(E0))) - target
			df = (a*b/2)*(1 - e*np.cos(E))
			E -= f/df
	
		theta_end = self.true_from_eccentric(E)
	
		# normalize to [0, 2π)
		theta_end = theta_end % (2*np.pi)
	
		return theta0 % (2*np.pi), theta_end


class arc(circle):
	def __init__(self, center, radius, start, end): #start is start angle and end is end angle
		super().__init__(center, radius)
		self.start = start
		self.end = end

	@property
	def theta(self):
		return ((self.end - self.start + math.pi) % (2*math.pi)) - math.pi

	def get_point01(self, t):
		t = npx.clamp01(t)
		return self.get_point(self.start + self.theta * t)

	def d(self):
		p0 = self.get_point01(0.)
		p1 = self.get_point01(1.)
	
		large_arc_flag = 1 if abs(self.theta) > math.pi else 0
		sweep_flag = 1 if self.theta > 0 else 0
	
		return (
			f"M {p0[0]} {p0[1]} "
			f"A {self.radius} {self.radius} 0 {large_arc_flag} {sweep_flag} {p1[0]} {p1[1]}"
		)

	def contains_angle(self, a):
		s = self.start
		t = self.theta
	
		if t >= 0:
			return 0 <= (a - s) % (2*math.pi) <= t
		else:
			return 0 >= (a - s) % (2*math.pi) >= t

	@property
	def aabb(self):
		c = self.center
		r = self.radius
	
		angles = [self.start, self.end]
	
		for i in range(4):
			a = math.pi/2. * i
			if self.contains_angle(a):
				angles.append(a)
	
		points = [self.get_point(a) for a in angles]
		
		return aabb(points)

	@aabb.setter
	def aabb(self, value):
		cur = self.aabb
	
		scale = value.size / cur.size
	
		# arcs must remain circular → use uniform scale
		s = min(scale)
	
		self.center = (self.center - cur.min) * s + value.min
		self.radius *= s


class line(points):	#start = self[0], end = self[1]
	
	@property
	def vector(self): return self[1] - self[0]
	
	@property
	def length(self): return np.linalg.norm(self.vector)

	@property
	def direction(self): return npx.normalize(self.vector)

	@property
	def midpoint(self): return np.mean(self, axis=0)

	@property #in XY-plane	#return left normal
	def normal(self): return np.array([-self.direction[1], self.direction[0]])

	@property	#angle of inclination, from x-axis
	def angle(self): return math.atan2(self.vector[1], self.vector[0])

	@property
	def perpendicular_bisector(self): return line([self.midpoint, self.midpoint + self.normal])
	
	def padding(self, left, right, relative=False):
		if relative:
			left *= self.length
			right *= self.length
		dir = self.direction
		return line([self[0] + dir * left, self[1] - dir * right])

	def expand(self, amount, relative=False): return self.padding(-amount, -amount, relative=relative)

	def resize(self, value, pivot=0.5):
		origin = npx.lerp(*self, pivot)
		dir = self.direction
		return line([origin - dir * value * (1 - pivot), origin + dir * value * pivot])
	
	def subdivide(self, n): return polyline(npx.subdivide(self[0], self[1], n+1), closed=False).edges

	def set_position(self, pivot, value):	#pivot is normalized and value is not normalized
		pivot = npx.lerp(*self, pivot)
		delta = value - pivot
		return line(self + delta)

	def coincide(a, b, tol=1e-12):
		"""Check if two N-dimensional lines coincide."""
		return all(point_on_line(a, x, tol) for x in b)

	def lerp(self, t): return npx.lerp(*self, t)

	def lerp_line(self, ts): return line([self.lerp(t) for t in ts])

class segment(line):
	pass

class chord(line):
	def __init__(self, start, end, theta):
		super().__init__(start, end)
		self.theta = theta

	@property
	def radius(self): return self.length / (2 * np.sin(self.theta / 2))

	@property
	def distance_to_center(self): return self.radius * np.cos(self.theta / 2)

	@property
	def distance_to_circumference(self): return self.radius - self.distance_to_center

	@property
	def center(self): return self.midpoint + self.normal * self.distance_to_center

	def to_circle(self): return circle(self.center, self.radius)

	def to_arc(self, dir=-1): #dir = -1 is counterclockwise and dir = 1 is clockwise
		circle = self.to_circle()
		start_angle = circle.angle_of(self.start)
		end_angle = circle.angle_of(self.end)
		if end_angle < start_angle and dir == -1:
			end_angle += 2 * np.pi
		elif end_angle > start_angle and dir == 1:
			start_angle += 2 * np.pi
			#end_angle -= 2 * np.pi
		return arc(self.center, self.radius, start_angle, end_angle)



def add_symmetrical_handles(vertices, handle_length=.1, closed=True):
	result = []
	if closed:
		for i in range(len(vertices)):
			u = vertices[i-1] - vertices[i]
			v = vertices[(i+1)%len(vertices)] - vertices[i]
			theta = npx.angle(u, v)
			alpha = (np.pi - theta) * 0.5
			result += [vertices[i] + npx.rotate(u, -alpha) * handle_length, vertices[i], vertices[i] + npx.rotate(v, alpha) * handle_length]
	else:
		result.append(vertices[0])
		for i in range(1, len(vertices) - 1):
			u = vertices[i-1] - vertices[i]
			v = vertices[(i+1)%len(vertices)] - vertices[i]
			theta = npx.angle(u, v)
			alpha = (np.pi - theta) * 0.5
			result += [vertices[i] + npx.rotate(u, -alpha) * handle_length, vertices[i], vertices[i] + npx.rotate(v, alpha) * handle_length]
		result.append(vertices[-1])
	result = lshift(result)
	return result


def truncate(vertices, t=.25):	#, closed=True):
	result = []
	for x in vertices.edges:
		result += [npx.lerp(x[0], x[1], float(t)), npx.lerp(x[0], x[1], 1-float(t))]
	if not vertices.closed:
		result[0] = vertices[0]
		result[-1] = vertices[-1]
	return polyline(result, closed=vertices.closed)

#returns the vertex indices of a polygon, a star polygon or a polygon compound, is denoted by its Schläfli symbol {p/q}, where p and q are relatively prime (they share no factors) and q ≥ 2
def polygram(p, q):	#p = total number of vertices, q = step size (how many points to skip when drawing)
	if q >= p / 2 or p < 3:
		return []
	#print(p, q)
	if math.gcd(p, q) == 1: #returns a regular polygon or a regular star polygon
		return [[(i * q) % p for i in range(p)]]
	#else math.gcd(p, q) > 1: returns a regular polygon compound
	g = math.gcd(p, q)
	return [[(j + i * q) % p for i in range(p // g)] for j in range(g)]

#a regular polygon or a regular star polygon. #A regular polygram, as a general regular polygon, is denoted by its Schläfli symbol {p/q}, where p and q are relatively prime (they share no factors) and q ≥ 2
def regular_polygram(p, q, r=1.0, center=np.zeros(2), start=0.0):	#p = total number of vertices on the circle, q = step size (how many points to skip when drawing)
	return Mesh(npx.on_circle(n=p, r=r, center=center, start=start), polygram(p, q))

def star(n, r=1.0, spoke_ratio=0.5, t=0.5, center=np.zeros(2), start=0.0):
	vertices = npx.on_circle(n=n, r=r, center=center, start=start)
	midpoints = npx.on_circle(n=n, r=r * spoke_ratio, center=center, start=start + (math.pi * 2.0 * t) / float(n))
	result = []
	for i in range(n):
		result += [vertices[i], midpoints[i]]
	return result

def gear(n, r=1.0, spoke_ratio=0.5, t=0.25, u=0.75, center=np.zeros(2), start=0.0):
	points = star(n=n, r=r, spoke_ratio=spoke_ratio, center=center, start=start)
	result = []
	for i in range(0, len(points), 2):
		result += [
			npx.lerp(points[i], points[(i - 1) % len(points)], t),
			npx.lerp(points[i], points[i + 1], t),
			npx.lerp(points[i + 1], points[i], t),
			npx.lerp(points[i + 1], points[(i + 2) % len(points)], t),
		]
	return result

class radar_chart:
	def __init__(self, axis_count, step_count, radius, center=np.zeros(2)): #axes = spokes = radii
		self.axes = [[center, x] for x in npx.on_circle(axis_count, r=radius, center=center)]
		step = radius / step_count
		self.polygons = [npx.on_circle(axis_count, r=step * (i + 1), center=center) for i in range(step_count)]
	
	@property
	def axis_count(self): return len(self.axes)

	@property
	def step_count(self): return len(self.polygons)

	def data_point(self, axis, value): return npx.lerp(*self.axes[axis], float(value) / float(self.step_count))
	
	def data_polygon(self, values): return [self.data_point(i, x) for i, x in enumerate(values)]


class Mesh():
	def __init__(self, vertices=None, indices=None, uvs=None):
		self.vertices = np.array(vertices if vertices is not None else [])
		self.indices = indices.copy() if indices is not None else []
		self.uvs = uvs.copy() if uvs is not None else []
		self.normals_interpolation = 'face_varying'	#constant (one value for entire primitive), face_varying (one value per face corner), vertex (one value per vertex), uniform (one value per face)
		self.uvs_interpolation = 'face_varying'
		self.colors = None
		self.colors_interpolation = 'face_varying'
		self.double_sided = False
	
	def face(self, i): return polyline(List.items(self.vertices, self.indices[i]), closed=True)

	@property
	def faces(self): return group([self.face(i) for i in range(len(self.indices))])

	@property
	def edges(self): return flatten([x.edges for x in self.faces])

	def edge_indices_of(self, face_index): return np.array([x for x in List.aranges(self.indices[face_index], 2, cycle=True)])

	@property
	def edge_indices(self): return np.array([self.edge_indices_of(i) for i in range(len(self.indices))])

	def to_graph(self):
		result = graph(*self.vertices)
		result.add_edges(flatten(self.edges_indices))
		return result

	def to_obj(self, path):
		lines = [f"v {x} {y} {z}" for x, y, z in self.vertices]
		if self.uvs is not None and len(self.uvs) > 0:
			lines += [f"vt {u} {v}" for u, v in self.uvs]
			lines += ["f " + " ".join(f"{i+1}/{i+1}" for i in x) for x in self.indices]
		else:
			lines += [f"f {' '.join(str(i + 1) for i in x)}" for x in self.indices]
		osx.write(path, '\n'.join(lines) + '\n')

	def from_obj(path):
		vertices = []
		uvs = []
		indices = []
		for line in osx.read(path).split('\n'):
			if line.startswith('v '):
				_, x, y, z = line.strip().split()
				vertices.append([float(x), float(y), float(z)])
			elif line.startswith('vt '):
				_, u, v = line.strip().split()
				uvs.append([float(u), float(v)])
			elif line.startswith('f '):
				parts = line.strip().split()[1:]
				f = []
				for p in parts:
					tokens = p.split('/')
					vertex_index = int(tokens[0]) - 1
					f.append(vertex_index)
					# Optional: If you want to store UV indices separately, you can do:
					# if len(tokens) > 1 and tokens[1]:
					#     uv_index = int(tokens[1]) - 1
					#     self.face_uv_indices.append(uv_index)
				indices.append(f)
		return Mesh(vertices, indices, uvs)
	
	def add_face(self, vertices):
		start_index = len(self.vertices)
		self.vertices.extend(vertices)
		self.indices.append(np.arange(start_index, start_index + len(vertices)))

	def add_faces(self, faces):
		for x in faces:
			self.add_face(x)

	def extrude(self, dir, face_index, flip=True):
		f = self.face(face_index)
		self.add_face([x + dir for x in f])
		for x in f.edges:
			self.add_face([x[0], x[1], x[1] + dir, x[0] + dir])
		if flip:
			for i in range(len(self.indices) - 5, len(self.indices)):
				self.flip_normal(i)

	def translate(self, vector):
		self.vertices = [x + vector for x in self.vertices]

	def scale(self, vector):
		self.vertices = [x * vector for x in self.vertices]

	@property
	def aabb(self): return aabb(self.vertices)	#bounds

	@aabb.setter
	def aabb(self, value):
		self.vertices = set_aabb(self.vertices, value)
	
	@property
	def pivot(self): return self.aabb.normalize_point(np.zeros(3))

	@pivot.setter	#pivot is normalized
	def pivot(self, value): self.translate(-self.aabb.denormalize_point(value))

	def merge(meshes):
		vertices = []
		indices = []
	
		offset = 0
	
		for mesh in meshes:
			indices.extend(
				[[offset + i for i in x] for x in mesh.indices]
			)
	
			vertices.append(mesh.vertices)
			offset += len(mesh.vertices)
	
		vertices = np.concatenate(vertices)
	
		return Mesh(vertices, indices)

	def make_double_sided(mesh):
		mesh.indices += [list(reversed(x)) for x in mesh.indices]

	def flip_normal(mesh, i):
		mesh.indices[i] = list(reversed(mesh.indices[i]))

	def flip_normals(mesh):
		mesh.indices = [list(reversed(x)) for x in mesh.indices]

	def make_vertices_unique(self):
		result = Mesh()
		for x in self.faces:
			result.add_face(*x)
		return result

	def face_normal(self, i):
		pts = self.face(i)
		n = np.zeros(3)

		for i in range(len(pts)):
			p0 = pts[i]
			p1 = pts[(i + 1) % len(pts)]

			n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
			n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
			n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])

		return npx.normalize(n)

	@property
	def face_normals(self):	#uniform
		return [self.face_normal(i) for i in range(len(self.indices))]

	@property
	def normal(self):	#constant, one normal for the entire mesh
		fn = self.face_normals.sum(axis=0)
		l = np.linalg.norm(fn)
		return fn / l if l != 0 else fn

	@property
	def vertex_normals(self):	#vertex
		vcount = len(self.vertices)
		acc = np.zeros((vcount, 3))

		fnormals = self.face_normals

		for fi, f in enumerate(self.indices):
			for vi in f:
				acc[vi] += fnormals[fi]

		# normalize
		lengths = np.linalg.norm(acc, axis=1)
		lengths[lengths == 0] = 1.0
		return acc / lengths[:, None]

	@property
	def flat_corner_normals(self):	#face varying flat
		fnormals = self.face_normals
		return np.asarray([
			fnormals[fi]
			for fi, f in enumerate(self.indices)
			for _ in f
		])

	@property
	def smooth_corner_normals(self):	#face varying smooth
		vnormals = self.vertex_normals
		return np.asarray([
			vnormals[vi]
			for f in self.indices
			for vi in f
		])


class angle(points):	#list):

	@property
	def rays(self): return [line([self[1], self[0]]), line([self[1], self[2]])]

	@property
	def vectors(self): return np.array([x.vector for x in self.rays])

	@property
	def size(self): return npx.angle(*self.vectors)

	@property
	def bisector(self): return npx.normalize(np.sum([x.direction for x in self.rays], axis=0))

	def to_arc(self, radius):
		v0 = npx.normalize(self.vectors[0])
		v1 = npx.normalize(self.vectors[1])
	
		start = math.atan2(v0[1], v0[0])
		end   = math.atan2(v1[1], v1[0])
	
		return arc(self[1], radius, start, end)

class polyline(points):

	def __new__(cls, input_array, closed=True):
		obj = super().__new__(cls, input_array, closed=closed)
		return obj

	@property
	def edges(p): return [line(x) for x in List.aranges(p, 2, cycle=p.closed)]
	
	def edge(p, index): return line([p[index], p[(index + 1) % len(p)]])
	
	def vertex_angle(p, index): return angle(List.arange(p, 3, start=index - 1))

	@property
	def vertex_angles(p): return [p.vertex_angle(i) for i in range(0 if p.closed else 1, len(p) - (0 if p.closed else 1))]

	@property
	def lengths(p): return np.array([x.length for x in p.edges])

	@property
	def perimeter(p): return sum(p.lengths())
	
	@property
	def dual(p): return polyline([x.midpoint for x in p.edges], closed=p.closed)	#midpoint polygon
	
	def point_from_proportion(self, t):
		p = self.perimeter
		a = 0.0
		for x in self.edges:
			b = a + x.length / p
			if a <= t and t <= b:
				return npx.lerp(x[0], x[1], (t - a) / (b - a))
			a = b
		return None

	def subdivide(p, n):
		result = []
		for x in p.edges:
			result += [npx.lerp(x[0], x[1], i / n) for i in range(n)]
		if not p.closed:
			result.append(p[-1])
		return polyline(result, closed=p.closed)

	def incident_edges(p, vertex):
		return [
			line(List.arange(p, 2, start=i, closed=True))
			for i in [vertex - 1, vertex]
			if p.closed or (0 <= i < len(p) - 1)
		]

	def neighbors(p, vertex):	#return vertex-adjacent vertices
		return [
			p[i % len(p)]
			for i in (vertex - 1, vertex + 1)
			if p.closed or (0 <= i < len(p))
		]
	
	"""def incident_edges(p, vertex):	
		edges = [line(List.arange(p, 2, start=vertex - 1)), line(List.arange(p, 2, start=vertex))]
		if not p.closed:
			if vertex == 0:
				return edges[1:]
			elif vertex == len(p) - 1:
				return edges[:-1]
		return edges
	
	def neighbors(v, i):	#return ith-vertex-adjacent vertices
		n = len(v)
		if v.closed:
			return [v[(i - 1) % n], v[(i + 1) % n]]
		if i == 0:
			return [v[1]] if n > 1 else []
		elif i == n - 1:
			return [v[n - 2]] if n > 1 else []
		else:
			return [v[i - 1], v[i + 1]]

	def tangents(v):
		n = len(v)
		result = []
		for i in range(n):
			if v.closed:
				t = v[(i + 1) % n] - v[(i - 1) % n]	# Índices com wrap-around
				#print(t)
			else:
				if i == 0:
					t = v[1] - v[0]
				elif i == n - 1:
					t = v[-1] - v[-2]
				else:
					t = (v[i+1] - v[i-1]) * 0.5
			result.append(npx.normalize(t))
		return result"""



	@property
	def normals(p): return np.array([x.normal for x in p.edges])

	@property
	def vertex_normals(p):
		return np.array([npx.normalize(np.sum([x.normal for x in p.incident_edges(i)], axis=0)) for i in range(len(p))])

	@property
	def tangents(p): return np.array([np.array([-x[1], x[0]]) for x in p.vertex_normals])

	@property
	def perpendicular_bisectors(p): return np.array([x.perpendicular_bisector for x in p.edges])

	def expand(self, amount):
		return self + self.vertex_normals * amount
		#return polyline([self[i] + x * float(amount) for i, x in enumerate(self.vertex_normals)], closed=self.closed)
	
	def internal_angle_sum(n): return math.pi * (n - 2)

	def circumcenter(p): return line_line_intersection(*p.perpendicular_bisectors[:2])

	"""@property
	def area(v):
		area = 0.0
		n = len(v)
		for i in range(n):
			x1, y1 = v[i]
			x2, y2 = v[(i + 1) % n]
			area += (x1 * y2 - x2 * y1)
		return area * 0.5	#abs(area) * 0.5, let it return negative numbers too, to triangulate works fine

	@property
	def centroid(p): return np.mean(p, axis = 0)

	def is_clockwise(polygon):
		sum = 0
		for i in range(len(polygon)):
			x1, y1 = polygon[i]
			x2, y2 = polygon[(i + 1) % len(polygon)]
			sum += (x2 - x1) * (y2 + y1)
		return sum > 0"""

	@property
	def shoelace(v): return sum([np.cross(*x) for x in v.edges])	#Shoelace formula
		
	@property
	def iscw(v): return v.shoelace < 0	#is clockwise

	@property
	def isacw(v): return v.shoelace > 0	#is anticlockwise

	@property
	def area(v): return 0.5 * v.shoelace	#If the polygon is negatively oriented, then the result is negative
	
	@property
	def centroid(v):

		A = v.area
	
		if abs(A) < 1e-12:
			return v.vertex_centroid
	
		s = np.zeros_like(v[0])

		for x in v.edges:
			s += np.sum(x, axis=0) * np.cross(*x)
	
		return s / (6 * A)
	
	def scale(self, factor, pivot=None):
		if pivot is None:
			pivot = np.zeros_like(self[0])
		result = copy.deepcopy(self)
		result[:] = pivot + (self - pivot) * factor
		return result
	
	def line_intersection(p, line): return [mat.segment_line_intersection(x, line) for x in p.edges]

	def ray_intersection(p, ray): return [mat.segment_ray_intersection(x, ray) for x in p.edges]

	def segment_intersection(p, seg): return [mat.segment_segment_intersection(x, seg) for x in p.edges]



	"""def contains_point(vertices, point):
		ray = [point, point + np.array([1e6, 0])]	# Make a ray starting from the point → far to the right (for odd-even test)
		hits = polyline.ray_intersection(vertices, ray, closed=True)	# Count intersections between the ray and polygon edges
		return len([h for h in hits if h is not None]) % 2 == 1	# Odd number of hits → point is inside"""

	def contains_point(polygon, point):
		x, y = point
		inside = False
		n = len(polygon)
		for i in range(n):
			x0, y0 = polygon[i]
			x1, y1 = polygon[(i + 1) % n]
			# Check if point is within y bounds of the edge and to the left of it
			if (y0 > y) != (y1 > y):
				x_intersect = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
				if x < x_intersect:
					inside = not inside
		return inside
	
	def triangulate(v):
		n = len(v)
		if n < 3:
			return []
	
		indices = list(range(n))
		if v.iscw:
			indices.reverse()
	
		triangles = []
	
		while len(indices) > 3:
			found_ear = False
			for i in range(len(indices)):
				ti = List.arange(indices, 3, start=i, closed=True)

				t = v[ti]
				if not t.isacw:
					continue
	
				ear_found = True
				for j in indices:
					if j in ti:
						continue
					if polyline.contains_point(t, v[j]):
						ear_found = False
						break
	
				if ear_found:
					triangles.append(t.copy())
					del indices[ti[1]]
					found_ear = True
					break
	
			if not found_ear:
				break  # Polígono pode estar com interseções ou ser degenerado
	
		if len(indices) == 3:
			triangles.append(indices)
	
		return triangles

	def to_stroke(v, width, align=0.5, join='miter', cap='butt'):	#join in ['bevel', 'butt', 'miter'], cap in ['butt', 'square']
		if v.closed:
			if np.allclose(v[0], v[-1]):
				v = v[:-1]
		else:
			if cap == 'square':
				v = polyline(v)
				v[0] += npx.normalize(v[0] - v[1]) * width * 0.5
				v[-1] += npx.normalize(v[-1] - v[-2]) * width * 0.5
		result = []
		if join == 'bevel':
			for x in v.edges:
				normal = polyline.normal(x, outward=True)
				result.append(x[0] + normal * width * (1.0 - align))
				result.append(x[1] + normal * width * (1.0 - align))
				result.insert(0, x[0] - normal * width * align)
				result.insert(0, x[1] - normal * width * align)
		elif join == 'butt':
			for i, x in enumerate(v.vertex_normals):
				result.append(v[i] + x * width * (1.0 - align))
				result.insert(0, v[i] - x * width * align)
		elif join == 'miter':
			angles = [x.size() for x in polyline.vertex_angles(v, closed=closed)]
			if not v.closed:
				angles = [math.pi] + angles + [math.pi]
			for i, x in enumerate(v.vertex_normals):
				s = math.sin(angles[i] / 2)
				result.append(v[i] + x * (width * (1.0 - align)) / s)
				result.insert(0, v[i] - (x * width * align) / s)
		if v.closed:
				mid = len(result) // 2
				#print(result, mid, result[mid - 1])
				return polyline([result[mid - 1]] + result + [result[mid]])
		else: return polyline(result)

	def rotate_around(p, angle, center=np.zeros(2)): return polyline([npx.rotate_around(x, angle, center) for x in p], closed=p.closed)

	"""def from_vectors(v, axis=None, start=np.zeros(2)):	#concatenation of n vectors (or edges) end-to-end starting from start
		result = polyline(v)
		if axis is None:
			result[0] += start
		else:
			result[0][axis] += start[axis]
		for i in range(1, len(result)):
			if axis is None:
				result[i] += result[i - 1]
			else:
				result[i][axis] += result[i - 1][axis]
		#print(v, result)
		return result
	
	def to_vectors(v, axis=None, start=np.zeros(2)):
		result = polyline(v)
		if axis is None:
			result[0] -= start
		else:
			result[0][axis] -= start[axis]
		for i in range(1, len(result)):
			if axis is None:
				result[i] -= result[i - 1]
			else:
				result[i][axis] -= result[i - 1][axis]
		#print(v, result)
		return result"""

	def from_vectors(v, axis=None, start=np.zeros(2)):	#concatenation of n vectors (or edges) end-to-end starting from start
		result = np.array(v, copy=True)
	
		if axis is None:
			result = np.cumsum(result, axis=0)
			result += start
		else:
			result[:, axis] = np.cumsum(result[:, axis])
			result[:, axis] += start[axis]
	
		return polyline(result)

	def to_vectors(v, axis=None, start=np.zeros(2)):
		result = np.empty_like(v)
	
		if axis is None:
			result[0] = v[0] - start
			result[1:] = v[1:] - v[:-1]
		else:
			result[:] = v
			result[0, axis] -= start[axis]
			result[1:, axis] -= v[:-1, axis]
	
		return polyline(result)
	
	"""def clip(p1, p2):	#split the edges of polyline p1 wherever they intersect edges of polyline p2
		result = []
		for edge in p1.edges():
			result.append(edge[0])
			inter = [
				p for y in p2.edges()
				if (p := segment_segment_intersection(edge, y)) is not None	#mat.segment_segment_intersection(edge, y)) is not None
				and not any(np.array_equal(p, v) for v in edge)
			]
			if inter:
				result.extend(sort_by_distance(inter, edge[0]))
		return polyline(result, closed=p1.closed)"""

	def clip(p1, p2):  # split the edges of polyline p1 wherever they intersect edges of polyline p2
		result = []
		for edge in p1.edges:
			result.append(edge[0])
			points = []
			for y in p2.edges:
				inter = segment_segment_intersection(edge, y)
				if inter is None:
					continue
				# Flatten: if it's a segment (2 points), treat both as intersection points
				if isinstance(inter, np.ndarray) and inter.shape[0] == 2 and inter.ndim == 2:
					points.extend(inter)
				else:
					points.append(inter)
			# Ignore points equal to endpoints of current edge
			points = [p for p in points if not any(np.array_equal(p, v) for v in edge)]
			# Sort by distance from edge[0] and append
			if points:
				result.extend(sort_by_distance(points, edge[0]))
		return polyline(result, closed=p1.closed)

	

	def simplify(self):	# remove points that lie exactly on the line between their neighbors
		result = [x for i, x in enumerate(self) if not point_on_segment(self.neighbors(i), x)]
		#print(len(result))
		return result


class polygon:
	def internal_angle(n): return polyline.internal_angle_sum(n) / n

	#a = apothem, r = radius, s = side
	def r2a(n): return math.cos(math.pi / n)
	def r2s(n): return 2 * math.sin(math.pi / n)
	def a2s(n): return 2 * math.tan(math.pi / n)


class triangle(polyline):
	@property
	def angle_bisectors(v): return group([line([x[1], line_line_intersection(line([x[1], x[1] + x.bisector]), line(x[[0, 2]]))]) for x in v.vertex_angles])

	@property
	def altitudes(v): return group([line([x[1], project_point_on_line(x[1], line(x[[0, 2]]))]) for x in v.vertex_angles])

	@property
	def medians(v): return group([line([x[1], line(x[[0, 2]]).midpoint]) for x in v.vertex_angles])

	@property
	def incenter(v): return line_line_intersection(*v.angle_bisectors[:2])

	@property
	def orthocenter(v): return line_line_intersection(*v.altitudes[:2])



class bezier(points):
	
	def get_point(p, t):
		# Recursive Bézier evaluation (De Casteljau's algorithm)
		if len(p) == 1:
			return p[0]
		else:
			return (1 - t) * bezier.get_point(p[:-1], t) + t * bezier.get_point(p[1:], t)

		"""#Evaluate a Bézier curve at parameter t using Bernstein basis.
		n = len(points) - 1
		points = np.array(points)
		point = np.zeros_like(points[0])
		for i in range(n + 1):
			binom = np.math.comb(n, i)
			point += binom * ((1 - t) ** (n - i)) * (t ** i) * points[i]
		return point"""
	
	def get_derivative(p, t):
		# Derivative of Bézier curve (based on differences between control points)
		n = len(p) - 1
		if n < 1:
			raise ValueError("Need at least two points for a derivative")
		derivative_points = [n * (p[i + 1] - p[i]) for i in range(n)]
		return get_point(t, *derivative_points)

	def sample(p, steps): return polyline([p.get_point(t) for t in np.linspace(0.0, 1.0, steps)], closed=False)

	def length(p, resolution=100): return p.sample(resolution).perimeter(closed=False)

	def sample_by_size(p, size, resolution=100):
		length = p.length(resolution)
		steps = max(2, int(math.ceil(length / size)))	#minimum 2 samples
		#print(length, size, steps)
		return p.sample(steps)

class polybezier(points):	#composite Bézier curve or Bézier spline

	def __new__(cls, input_array, endpoints=None, closed=False):
		obj = super().__new__(cls, input_array, closed=closed, endpoints=list(range(len(input_array))) if endpoints is None else endpoints)
		return obj
	
	def get_endpoints(self): return polyline([self[i] for i in self.endpoints], closed=self.closed)
	
	@property
	def curves(self):
		result = []
		for i in range(len(self.endpoints) - 1):
			result.append(self[self.endpoints[i] : self.endpoints[i+1]+1])
		# last curve
		if self.closed:
			result.append(np.concatenate([self[self.endpoints[-1]:], self[:self.endpoints[0]]]))
		return [bezier(x) for x in result]

	def sample_by_size(p, size, resolution=100):
		curves = [x.sample_by_size(size, resolution=resolution) for x in p.curves]
		parts = []
		for c in curves:
			#print(c)
			parts.append(c[:-1])
		if not p.closed:	# add last point if open path
			parts.append(curves[-1][-1:])
		return polyline(np.vstack(parts), closed=p.closed)
	
	def d_coordinates(self, closed=True):
		result = []
		result.append(self[self.endpoints[0]:self.endpoints[0]+1])	# first command parameters
		for i in range(len(self.endpoints) - 1):	# middle command parameters
			result.append(self[self.endpoints[i]+1 : self.endpoints[i+1]+1])
		# last command parameters
		if closed:
			result.append(np.concatenate([self[self.endpoints[-1] + 1:], self[:self.endpoints[0] + 1]]))
		return result

	def d(self):
		d = ''
		for x in self.d_coordinates():
			d += 'MQC'[len(x) - 1]
			d += ' ' + ' '.join([str(y) for y in x.flatten()]) + ' '
		#print(d)
		return d

def smooth_control_points(p, index, extents=10):
	angle = p.vertex_angle(index)
	rot = (math.pi - angle.size()) * 0.5
	cp = npx.normalize(npx.rotate(angle.vectors()[0], rot)) * extents
	cp2 = npx.normalize(npx.rotate(angle.vectors()[1], -rot)) * extents	
	return [cp, cp2]

def smooth_polyline(p, extents=10, closed=True):
	points = []
	endpoints = []
	cps = [smooth_control_points(p, i, extents=extents) for i in range(len(p))]
	for i in range(len(p) - (0 if closed else 1)):
		j = (i + 1) % len(p)
		points += [p[i], cps[i][1] + p[i], cps[j][0] + p[j]]
		endpoints += [3 * i]
	if not closed:
		points.append(p[-1])
		endpoints.append(3 * (len(p) - 1))
	return bezier(points, endpoints)


class tspan:
	def __init__(self, inner_text, position, font='arial.ttf', font_size=12, pivot=np.ones(2) * 0.5):
		self.inner_text = str(inner_text)
		self.position = position
		self.font = font
		self.font_size = font_size
		self.pivot = pivot

	@property
	def size(self):
		return PILx.get_size(self.inner_text, self.font, self.font_size)

	@property
	def aabb(self):
		#print(self.size)
		return rect(np.zeros(2), self.size).set_position(self.pivot, self.position)

	@aabb.setter
	def aabb(self, value):
		self.position = value.denormalize_point(self.pivot)

class text(tspan):
	def __init__(self, inner_text, position, font='arial.ttf', font_size=12, pivot=np.ones(2) * 0.5, line_spacing=4, align=.5):
		super().__init__(inner_text, position, font, font_size, pivot)
		self.line_spacing = line_spacing
		self.align = .5

	@property
	def size(self):
		res = PILx.getsize(self.inner_text.split('\n'), self.font, self.font_size, self.line_spacing)
		#print(res)
		return res

	@property
	def lines(self):
		result = []
		y = 0
		for i, x in enumerate(self.inner_text.split('\n')):
			#pos = self.position + npx.ei(1, 2) * y
			pos = np.array([self.aabb.denormalize_point_component(self.align, 0), self.aabb.min[1] + y])
			line = tspan(x, pos, self.font, self.font_size, np.array([self.align, 0.]))	#self.pivot)
			y += line.size[1] + self.line_spacing
			result.append(line)
		return result








def prism_laterals(count, start_index1=0, start_index2=None): #closed
	start_index2 = start_index1 + count if start_index2 is None else start_index2
	return [[start_index1 + i, start_index1 + (i + 1) % count, start_index2 + (i + 1) % count, start_index2 + i] for i in range(count)]

def pyramid_laterals(count, start_index=0, apex=None):
	apex = start_index + count if apex is None else apex
	return [[start_index + i, start_index + (i + 1) % count, apex] for i in range(count)]

def antiprism_laterals(count, start_index1=0, start_index2=None): #closed
	start_index2 = start_index1 + count if start_index2 is None else start_index2
	result = [[start_index1 + i, start_index1 + (i + 1) % count, start_index2 + i] for i in range(count)]
	result += [[start_index1 + (i + 1) % count, start_index2 + (i + 1) % count, start_index2 + i] for i in range(count)]
	return result


from numbers import Number

def enlongated(*primitives, r=1, height=1, gyro=False, start_angle=None):
	if isinstance(height, Number):
		height = [height] * (len(primitives) - 1)
	start_angle = [0] * len(primitives) if start_angle is None else start_angle
	vertices = []
	for i, x in enumerate(primitives):
		if x > 1:
			vertices += [np.array([y[0], sum(height[:i], start=0), y[1]]) for y in npx.on_circle(x, r=r, start=(math.pi/x) * start_angle[i])]
		else:
			vertices.append(np.array([0, sum(height[:i], start=0), 0]))
	return vertices, enlongated_faces(*primitives, gyro=gyro)

def enlongated_faces(*primitives, gyro=False): #primitives contains only n or 1
	result = []
	if primitives[0] > 1:
		result.append(list(range(primitives[0])))
	start_index = 0
	for i in range(len(primitives) - 1):
		m, n = primitives[i], primitives[i+1]
		if m > 1 and n > 1:
			result += antiprism_laterals(m, start_index) if gyro else prism_laterals(m, start_index)
		elif m == 1:
			result += pyramid_laterals(n, start_index + 1, start_index)
		else:
			result += pyramid_laterals(m, start_index, start_index + m)
		start_index += m
	if primitives[-1] > 1:
		result.append(list(range(start_index, start_index + primitives[-1])))
	#print(result)
	return result

def prism(count, r=1, height=1): return enlongated(count, count, r=r, height=height)

def pyramid(count, r=1, height=1): return enlongated(count, 1, r=r, height=height)

def bipyramid(count, r=1, height=1): return enlongated(1, count, 1, r=r, height=height)

def antiprism(count, r=1, height=1): return enlongated(count, count, r=r, height=height, gyro=True, start_angle=[0,1])

def elongated_pyramid(count, r=1, height=1): return enlongated(count, count, 1, r=r, height=height)

def elongated_bipyramid(count, r=1, height=1): return enlongated(1, count, count, 1, r=r, height=height)

def gyroelongated_pyramid(count, r=1, height=1): return enlongated(count, count, 1, r=r, height=height, gyro=True, start_angle=[0,1,0])

def gyroelongated_bipyramid(count, r=1, height=1): return enlongated(1, count, count, 1, r=r, height=height, gyro=True, start_angle=[0,0,1,0])

def ring(count, r=1, R=2, height=1):
	vertices = [np.array([pt[0], h, pt[1]]) for radius, h in product([r, R], [0, height]) for pt in npx.on_circle(count, r=radius)]
	indices = []
	for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
		indices += prism_laterals(count, start_index1=count * a, start_index2=count * b)
	return vertices, indices

def randomize2(vertices, r=.1): return [npx.random_in_circle(r) + x for x in vertices]

def randomize3(vertices, r=.1): return [npx.random_in_sphere(r) + x for x in vertices]


def conic_sort(edges):
	return sorted(edges, key=lambda e: npx.angle2(npx.ei(0.0, 2), e.vector))

def sort_by_angle(self, clockwise=False):
	v = np.asarray(self)
	c = v.mean(axis=0)
	a = np.arctan2(v[:,1] - c[1], v[:,0] - c[0])
	i = np.argsort(a)
	return v[i[::-1] if clockwise else i]

def sort_by_distance(self, point, reverse=False):
	p = np.asarray(self)
	d = ((p - point)**2).sum(1)
	i = np.argsort(d)
	return p[i[::-1] if reverse else i]




def incident_edges(point, edges, eps=1e-9):
	return [e for e in edges if any(np.linalg.norm(v - point) <= eps for v in e)]

def angle_vector_plane(v, p1, p2):	#p1 and p2 are vectors that define the plane
	n = np.cross(p1, p2)	# Plane normal via cross product
	v_norm = v / np.linalg.norm(v)	# Normalize normal and vector
	n_norm = n / np.linalg.norm(n)
	angle_to_normal = np.arccos(np.clip(np.dot(v_norm, n_norm), -1.0, 1.0))	# Angle between v and plane normal (in radians)
	angle_to_plane = np.pi / 2 - angle_to_normal	# Angle between vector and plane
	return angle_to_plane	# return in radians


class group(list):
	@property
	def aabb(self):
		return aabb([x.aabb for x in self])

	@aabb.setter
	def aabb(self, value):
		cur = self.aabb
		for x in self:
			x.aabb = value.denormalize_rect(cur.normalize_rect(x.aabb))

	@property
	def local_aabb(self): return aabb([x.local_aabb for x in self])

	@property
	def global_aabb(self): return aabb([x.global_aabb for x in self])


def distribute(arr, axis=0, align=.5, gap=0.0):
	for i in range(1, len(arr)):
		pos = arr[i - 1].aabb.denormalize_point(np.array([1, float(align)])[[axis, 1 - axis]]) + npx.ei(axis, 2) * float(gap)
		arr[i].aabb = arr[i].aabb.set_position(pivot = np.array([0, float(align)])[[axis, 1 - axis]], value = pos)

def rects(offset, sizes, axis=0, align=0.5, gap=0.0):
	dim = len(sizes[0])
	result = group([rect(np.zeros(dim), x) for x in sizes])
	result[0].min = offset
	#print(offset)
	distribute(result, axis=axis, align=align, gap=gap)
	return result


# ------------------- Point-on checks -------------------
def point_on_line(line, point, tol=1e-12):
    a, b = line
    return npx.collinear(b - a, point - a, tol)

def point_on_ray(ray, point, tol=1e-12):
    a, b = ray
    v = b - a
    return npx.collinear(v, point - a, tol) and np.dot(point - a, v) >= -tol

def point_on_segment(segment, point, tol=1e-12):
    a, b = segment
    v = b - a
    return npx.collinear(v, point - a, tol) and np.dot(point - a, v) >= -tol and np.dot(point - b, v) <= tol


def circle_circle_intersection(c0, c1, tol=1e-9):
	"""
	Compute intersection points between two circle objects.

	Returns:
		None  -> no intersection
		[]    -> coincident circles (infinite intersections)
		[p]   -> tangent (one point)
		[p1, p2] -> two intersection points (numpy arrays)
	"""

	p0 = c0.center
	p1 = c1.center
	r0 = c0.radius
	r1 = c1.radius

	d = np.linalg.norm(p1 - p0)

	# No intersection
	if d > r0 + r1 + tol:
		return None

	# One inside the other
	if d < abs(r0 - r1) - tol:
		return None

	# Coincident circles
	if d < tol and abs(r0 - r1) < tol:
		return []

	# Distance from p0 to midpoint
	a = (r0**2 - r1**2 + d**2) / (2*d)

	# Height from midpoint to intersections
	h_sq = r0**2 - a**2
	h_sq = max(h_sq, 0.0)
	h = np.sqrt(h_sq)

	# Midpoint
	midpoint = p0 + a * (p1 - p0) / d

	# Perpendicular direction
	perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d

	p_int1 = midpoint + h * perp
	p_int2 = midpoint - h * perp

	if h < tol:
		return [p_int1]  # tangent

	return [p_int1, p_int2]


def circle_line_intersection(c, l, tol=1e-9):
	"""
	Intersection between circle and line segment.

	c : circle instance
	l   : line instance (shape (2,2))

	Returns:
		None
		[p]
		[p1, p2]
	"""

	C = c.center
	r = c.radius

	P0 = np.array(l[0], dtype=float)
	P1 = np.array(l[1], dtype=float)

	d = P1 - P0          # direction vector
	f = P0 - C           # vector from circle center to line start

	a = np.dot(d, d)
	b = 2 * np.dot(f, d)
	c = np.dot(f, f) - r**2

	discriminant = b**2 - 4*a*c

	if discriminant < -tol:
		return None

	discriminant = max(discriminant, 0.0)
	sqrt_disc = np.sqrt(discriminant)

	t1 = (-b - sqrt_disc) / (2*a)
	t2 = (-b + sqrt_disc) / (2*a)

	points = []

	# Check if intersection is within segment [0,1]
	if 0 - tol <= t1 <= 1 + tol:
		points.append(P0 + t1 * d)

	if 0 - tol <= t2 <= 1 + tol and discriminant > tol:
		points.append(P0 + t2 * d)

	if not points:
		return None

	return points


def line_line_intersection(a, b, tol=1e-12):
    """
    Compute intersection of two N-dimensional lines.
    Returns:
        - copy of a if lines coincide
        - None if lines are parallel but separate
        - intersection point if lines intersect at a single point
    """
    if line.coincide(a, b, tol):
        return a.copy()  # lines are the same

    if npx.collinear(a.vector, b.vector, tol):
        return None  # parallel but separate

    # Solve a + t * va = b + s * vb
    va = np.asarray(a.vector)
    vb = np.asarray(b.vector)
    p1 = np.asarray(a[0])
    p2 = np.asarray(b[0])

    # Build matrix to solve: t*va - s*vb = p2 - p1
    A = np.column_stack((va, -vb))
    bvec = p2 - p1

    # Use least squares (works in any dimension)
    sol, residuals, rank, _ = np.linalg.lstsq(A, bvec, rcond=None)

    # If residual is large, no intersection
    if residuals.size > 0 and residuals[0] > tol:
        return None

    t = sol[0]
    intersection_point = p1 + t * va
    return intersection_point

# ------------------- Generic clip function -------------------
def clip_coincident(base, points, check_func):
    """Return overlapping part of coincident objects (line/ray/segment)."""
    inside = [p for p in points if check_func(base, p)]
    if not inside:
        return None
    # Single point or multiple points
    return inside[0] if len(inside) == 1 else np.array([min(inside, key=lambda x: tuple(x)),
                                                       max(inside, key=lambda x: tuple(x))])

# ------------------- Generic intersection function -------------------
def line_subsets_intersection(obj1, obj2, point_check1, point_check2=None, tol=1e-12):
    """
    Generic intersection between any two 1D objects (line, ray, segment)
    point_check1: check function for obj1
    point_check2: check function for obj2 (defaults to point_check1)
    """
    if point_check2 is None:
        point_check2 = point_check1

    inter = line_line_intersection(obj1, obj2, tol)
    if inter is None:
        return None

    # Coincident case
    if isinstance(inter, np.ndarray) and inter.shape == np.asarray(obj1).shape:
        return clip_coincident(obj1, obj2, point_check1)

    # Single point case
    if point_check1(obj1, inter) and point_check2(obj2, inter):
        return inter
    return None

# ------------------- Specific intersections -------------------
line_ray_intersection     = lambda line, ray, tol=1e-12: line_subsets_intersection(line, ray, point_on_ray)
line_segment_intersection = lambda line, seg, tol=1e-12: line_subsets_intersection(line, seg, point_on_segment)
ray_ray_intersection      = lambda r1, r2, tol=1e-12: line_subsets_intersection(r1, r2, point_on_ray)
ray_segment_intersection  = lambda ray, seg, tol=1e-12: line_subsets_intersection(ray, seg, point_on_ray, point_on_segment)
segment_segment_intersection = lambda s1, s2, tol=1e-12: line_subsets_intersection(s1, s2, point_on_segment)

def project_point_on_line(p, l):	#Project point p onto the line defined by points a and b.
	return l[0] + npx.project(p - l[0], l.vector)

def project_point_on_circle(p, c):
	"""Project point p onto hypersphere c."""
	return c.center + npx.normalize(p - c.center) * c.radius


def point_circle_delta(p, c):
	"""Signed distance from point to circle surface."""
	return npx.distance(p, c.center) - c.radius


def point_circle_distance(p, c):
	"""Distance from point to circle (0 if inside)."""
	return max(point_circle_delta(p, c), 0.)



def point_line_distance(p, l): return np.linalg.norm(project_point_on_line(p, l) - p)

def point_segment_closest_point(p, l):
	t = npx.inverse_lerp(*l, p)
	return npx.lerp(*l, np.clip(t, 0, 1))

def point_segment_distance(p, l): return np.linalg.norm(point_segment_closest_point(p, l) - p)

def circle_line_delta(c, l): return point_line_distance(c.center, l) - c.radius
"""Subtracting the circle’s radius gives the signed distance from the circle’s edge to the line:
	If the result is positive, the line is outside the circle.
	If the result is zero, the line is tangent to the circle.
	If the result is negative, the line intersects the circle."""

def circle_line_distance(c, l): return max(circle_line_delta(c, l), 0.)

def circle_segment_delta(c, l): return point_segment_distance(c.center, l) - c.radius

def circle_segment_distance(c, l): return max(circle_segment_delta(c, l), 0.)

def circle_polyline_distance(c, v):
	return min([circle_segment_distance(c, x) for x in v.edges])

def circle_rect_distance(c, rect):
	return circle_polyline_distance(c, rect2.corners(rect))







def circle_contains_point(c, p):
	"""Return True if point is inside or on circle."""
	return point_circle_delta(p, c) <= 0


def circle_circle_delta(a, b):
	"""Signed distance between circle surfaces."""
	return npx.distance(a.center, b.center) - (a.radius + b.radius)


def circle_circle_distance(a, b):
	"""Distance between circles (0 if overlapping)."""
	return max(circle_circle_delta(a, b), 0.0)


def circle_intersects_circle(a, b):
	"""Return True if circles overlap."""
	return circle_circle_delta(a, b) < 0


def circle_tangents_circle(a, b, tol=1e-12):
	"""Return True if circles are tangent."""
	return abs(circle_circle_delta(a, b)) < tol


def circle_circle_shortest_segment(a, b):
	"""Shortest segment between two circles."""
	p1 = project_point_on_circle(b.center, a)
	p2 = project_point_on_circle(a.center, b)
	return geo.line([p1, p2])

def segments_shape_intersection(segments, other, func):
	"""
	Compute intersections between a list of segments and another object.
	func must be a segment-shape intersection function.
	"""
	result = []
	for s in segments:
		inter = func(other, s)
		if inter is not None:
			result.append(inter)
	return result


def boundary_line_intersection(polygon, line):
	return segments_shape_intersection(polygon.edges, line, line_segment_intersection)


def boundary_ray_intersection(polygon, ray):
	return segments_shape_intersection(polygon.edges, ray, ray_segment_intersection)


def boundary_segment_intersection(polygon, segment):
	return segments_shape_intersection(polygon.edges, segment, segment_segment_intersection)





def segment_contains_segment(a, b):	# check if segment a contains segment b
	return all(point_on_segment(a, x) for x in b)

def arrow_contains_arrow(a, b):	# check if arrow a contains arrow b
	if segment_contains_segment(a, b):
		return np.array_equal(line(a).direction, line(b).direction)
	return False


def boundary_contains_point(p, point): return any(point_on_segment(e, point) for e in p.edges)

def interior_contains_point(p, point): return p.contains_point(point) and not boundary_contains_point(p, point)

def polygon_contains_point(p, point): return p.contains_point(point) or boundary_contains_point(p, point)


def boundary_contains_points(p, points): return all(boundary_contains_point(p, pt) for pt in points)

def interior_contains_points(p, points): return all(interior_contains_point(p, pt) for pt in points)

def polygon_contains_points(p, points): return all(polygon_contains_point(p, pt) for pt in points)


def boundary_contains_segment(p, s): return any(segment_contains_segment(x, s) for x in p.edges)

def interior_contains_segment(p, s):
	if not interior_contains_points(p, s):
		return False
	return not any(segment_segment_intersection(e, s) for e in p.edges)

def polygon_contains_segment(p, s): return boundary_contains_segment(p, s) or interior_contains_segment(p, s)


def pairwise_distances(points: np.ndarray) -> np.ndarray:
	"""
	Compute sorted list of all pairwise distances
	"""
	n = len(points)
	dists = []

	for i in range(n):
		for j in range(i + 1, n):
			d = np.linalg.norm(points[i] - points[j])
			dists.append(d)

	return np.sort(np.array(dists))


def congruent(A: np.ndarray, B: np.ndarray, tol=1e-6) -> bool:
	"""
	Check if two point sets are congruent (isometric)
	"""
	if A.shape != B.shape:
		return False

	dA = pairwise_distances(A)
	dB = pairwise_distances(B)

	return np.allclose(dA, dB, atol=tol)


def similar(A: np.ndarray, B: np.ndarray, tol=1e-6) -> bool:
	"""
	Check if two point sets are similar (up to scaling)
	"""
	if A.shape != B.shape:
		return False

	dA = pairwise_distances(A)
	dB = pairwise_distances(B)

	# Avoid division by zero
	if np.any(dA == 0):
		return False

	ratios = dB / dA

	return np.allclose(ratios, ratios[0], atol=tol)






#ADD SHAPE SVG DRAW METHODS
import pyx.generic.generic as generic


shapes = [polybezier, circle, ellipse, line, polyline, rect, group, arc, tspan, text, Transform]
for x in shapes:
	x.set = lambda self, **attrib: generic.set(self, attrib={**getattr(self, "attrib", {}), **attrib})
	x.get = lambda self, *keys: [self.attrib[k] for k in keys] if hasattr(self, 'attrib') else None



def findall_at_distance(point, others, dist, tol=1e-6, metric=lambda x, y: np.linalg.norm(x - y), mask=None):
	return find_indices(others, lambda x: abs(metric(point, x) - dist) < tol, mask=mask)

def findall_in_distance(point, others, dist, tol=1e-6, metric=lambda x, y: np.linalg.norm(x - y), mask=None):
	return find_indices(others, lambda x: metric(point, x) <= dist + tol, mask=mask)


