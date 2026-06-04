import math
import cmath
from multipledispatch import dispatch
from numbers import Number
import numpy as np
import itertools
from pyx.collectionsx import List as ls
import random
from pyx.numpyx_geo import polyline, line
from itertools import product

def modular_distance(a, b, n):
	d = abs(a - b) % n
	return min(d, n - d)

def slice_by_size(arr, size):	#Divide a 1D array into subarrays of given size. Last chunk may be smaller.
	return [arr[i:i+size] for i in range(0, len(arr), size)]

def slice_by_count(arr, count):
	return slice_by_size(arr, math.ceil(len(arr) / count))

def project(v, u): return (np.dot(v, u) / np.dot(u, u)) * u	#Project vector v onto vector u.

def fill(obj, length, filler=0):	#put it as a methodclass in a class called points that bases line, polyline, bezier, path and so on...
	return np.array([np.append(x, [filler] * (length - len(x))) for x in obj])

#works for both scalars (numbers) and NumPy arrays
def clamp(value, min, max): return np.minimum(np.maximum(value, min), max)

def clamp01(value): return clamp(value, 0.0, 1.0)

def repeat(t, length):
	return clamp(t - math.floor(t / length) * length, 0.0, length)

def ping_pong(t, length):
	t = repeat(t, length * 2)
	return length - abs(t - length)

def clamp_magnitude(v, max_magnitude):
	v = np.asarray(v, dtype=float)
	mag = np.linalg.norm(v)

	if mag == 0:
		return v

	if mag > max_magnitude:
		return v * (max_magnitude / mag)

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

def min_max_normalize(arr):
	"""
	Normalize a 1D array to the [0, 1] range using min-max scaling.
    
	Parameters:
		arr (np.ndarray): Input 1D array.
        
	Returns:
		np.ndarray: Normalized array with values in [0, 1].
	"""
	#arr = np.asarray(arr)  # Ensure input is an ndarray
	a = arr.min()
	b = arr.max()
    
	if a == b:
		# Avoid division by zero if all elements are the same
		return np.zeros_like(arr, dtype=float)
    
	return (arr - a) / (b - a)



def raised_norm(a, n=2): return sum(abs(a)**n)	#default n=2 means sqr magnitude
def norm(a, n=2): return raised_norm(a, n) ** (1./n)	#default n=2 means magnitude
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

#@dispatch(np.ndarray, np.ndarray)
#def random_range(start, stop): return np.array(list(map(lambda x, y: random_range(x, y), start, stop)))

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





def affine_transform(M, arr):
	"""
	Applies an (n+1)x(n+1) affine transform to points.

	M   : (n+1, n+1) affine matrix
	arr : (N, n) array of points OR (n,) single point
	"""
	M = np.asarray(M, dtype=float)
	arr = np.asarray(arr, dtype=float)

	# Handle single point
	if arr.ndim == 1:
		arr = arr[None, :]

	if arr.ndim != 2:
		raise ValueError("arr must be shape (N, n) or (n,)")

	n = arr.shape[1]

	if M.shape != (n + 1, n + 1):
		raise ValueError(f"Expected {(n+1)}x{(n+1)} affine matrix")

	# Homogeneous coordinates
	ones = np.ones((arr.shape[0], 1))
	pts_h = np.hstack([arr, ones])      # (N, n+1)

	# Apply transform
	out = pts_h @ M.T                   # (N, n+1)

	return out[:, :n]



from typing import Union
#@dispatch(Number, Number, Number)

def contains(min, max, value) -> bool:
	if all(isinstance(x, Number) for x in [min, max, value]):
		return min <= value <= max
	#else is instance of Union[np.ndarray, list, tuple]
	if len(min) != len(max) or len(min) != len(value):
		raise ValueError("min, max, and value must have the same length")
	return all(min[i] <= x <= max[i] for i, x in enumerate(value))


#Vector functions. Works in any dimension.

def normalize(u, tol=1e-12):
	"""
	Return the unit vector of u. If ‖u‖ < tol, return a zero vector instead.
	"""
	#u = np.asarray(u)
	nu = np.linalg.norm(u)

	if nu < tol:
		return np.zeros_like(u)

	return u / nu

def collinear(u, v, tol=1e-12):
	"""
	Return True if vectors u and v are collinear (parallel or antiparallel), False otherwise.
	
	Two vectors are collinear if one is a scalar multiple of the other.
	This is equivalent to |u·v| = ||u|| * ||v|| or checking if normalize(u) == normalize(v)
	or normalize(u) == -normalize(v).
	"""
	nu = np.linalg.norm(u)
	nv = np.linalg.norm(v)

	if nu < tol or nv < tol:
		return True

	return abs(np.dot(u, v)) >= (1 - tol) * nu * nv

def rank(points):	#minimum dimension
	"""
	Rank 0: Coincident (Same Location)
	Rank 1: Collinear (Same Line)
	Rank 2: Coplanar (Same Plane)
	Rank k: Same k-flat
	"""
	points = np.array(points)
	if len(points) == 0:
		return 0
	# Shift points so that first point is the origin
	origin = points[0]
	shifted = points - origin
	# Compute the rank (dimension of the span of vectors)
	return np.linalg.matrix_rank(shifted)

def test_rank():
    test_cases = [
        # All points identical → dimension 0
        (np.array([[1, 1], [1, 1], [1, 1]]), 0),

        # Collinear points → dimension 1
        (np.array([[0, 0], [1, 1], [2, 2]]), 1),
        (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), 1),

        # Coplanar points not collinear → dimension 2
        (np.array([[0,0,0], [1,0,0], [0,1,0]]), 2),

        # Random 3D points → dimension 3
        (np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]), 3),

        # 4D points in a line → dimension 1
        (np.array([[0,0,0,0], [1,1,0,0], [2,2,0,0]]), 1),

        # Empty points → dimension 0
        (np.array([]), 0),
    ]

    for i, (points, expected) in enumerate(test_cases, 1):
        result = rank(points)
        assert result == expected, f"Test {i} failed: expected {expected}, got {result}"
        print(f"Test {i} passed: dimension = {result}")

# Run the tests
#test_rank()





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

class rand:
	def __init__(self, min, max):
		self.min = min
		self.max = max

	def __call__(self):
		raise NotImplementedError

	def sample(self, count):
		return [self() for _ in range(count)]

class rand_number(rand):
	def __float__(self): return float(self())

	def __int__(self): return int(self())

class rand_float(rand_number):
	def __call__(self):
		return np.random.uniform(self.min, self.max)

class rand_int(rand_number):
	def __call__(self):
		return np.random.randint(self.min, self.max)



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





def quadratic(a, b, c):
	if a == 0:
		raise ValueError("a must not be zero")

	delta = b*b - 4*a*c
	den = 2*a

	if delta > 0:
		sqrt_delta = math.sqrt(delta)
		return (
			(-b + sqrt_delta) / den,
			(-b - sqrt_delta) / den
		)

	if delta == 0:
		return (-b / den,)

	# delta < 0 → complex
	sqrt_delta = cmath.sqrt(delta)
	return (
		(-b + sqrt_delta) / den,
		(-b - sqrt_delta) / den
	)



class lsystem:	#L-system or Lindenmayer system	#str based
	def __init__(self, alphabet, start, rules):	#alphabet contains variables and constants
		self.alphabet = alphabet
		self.start = start	#axiom, initiator
		self.rules = rules	#production rules, productions

	def rewrite(self, steps=1, state=None):

		if state is None:
			state = self.start

		for _ in range(steps):

			new_state = []

			for symbol in state:

				if symbol in self.rules:
					replacement = self.rules[symbol]

					if isinstance(replacement, str):
						new_state.extend(replacement)
					else:
						new_state.extend(replacement)

				else:
					new_state.append(symbol)

			if isinstance(state, str):
				state = ''.join(new_state)
			else:
				state = type(state)(new_state)

		return state

class block_lsystem(lsystem):
	def rewrite(self, steps=1, state=None):	#String Rewriting System
		if state is None:
			state = self.start
		state = np.asarray(state)

		if steps <= 0:
			return state

		block_shape = next(iter(self.rules.values())).shape
		dim = len(block_shape)

		if any(r.shape != block_shape for r in self.rules.values()):
			raise ValueError('all replacement patterns must have the same shape')

		for _ in range(steps):

			new_shape = tuple(
				state.shape[i] * block_shape[i]
				for i in range(dim)
			)

			new_state = np.empty(new_shape, dtype=state.dtype)

			for index in np.ndindex(state.shape):

				symbol = state[index]

				if symbol not in self.rules:
					raise KeyError(f'no rule for symbol {symbol}')

				pattern = self.rules[symbol]

				slices = tuple(
					slice(
						index[i] * block_shape[i],
						(index[i] + 1) * block_shape[i]
					)
					for i in range(dim)
				)

				new_state[slices] = pattern

			state = new_state

		return state



def all_ndarrays(shape, number):
	n = np.prod(shape)
	return np.array(
		list(product(range(number), repeat=n))
	).reshape((-1,) + tuple(shape))
