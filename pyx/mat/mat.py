import pyx.numpyx as npx
import pyx.numpyx_geo as geo
from multipledispatch import dispatch
from numbers import Number
import numpy as np
import random
import math
from pyx.array_utility import index_of
from functools import reduce

def hyperop(a, n, b): #a[n]b = a[n-1](a[n](b-1)), n>=1 #Hyperoperation
    if n == 0: # Successor
        return b + 1
    elif n == 1: # Addition
        return a + b
    elif n == 2: # Multiplication
        return a * b
    elif n == 3: # Exponentiation
        return a ** b
    elif n > 3: # Tetration and beyond
        if b == 0:
            return 1  # Base case for n > 3
        return hyperop(a, n - 1, hyperop(a, n, b - 1))

def fitf(arr, total):
	scale = total / sum(arr)
	return [x * scale for x in arr]

def fiti(arr, total, distribute_remainder=False):
	result = [math.floor(x) for x in fitf(arr, total)]
	remainder = total - sum(result)
	if distribute_remainder:
		for i in range(remainder):
			result[i] += 1
	else:
		result[0] += remainder
	return result

def arg(function, iterables, return_type=tuple):
	#print(iterables)
	return return_type(map(function, *iterables))

def normalize(values): return np.array(values) / max(values)

def normalize2(values): return [npx.inverse_lerp(min(values), max(values), x) for x in values]

def weights(values): return np.array(values) / sum(values)

def sgn(value): return -1 if value < 0 else 1 if value > 0 else 0

def lcm(a, b): return abs(a*b) // math.gcd(int(a), int(b))

@dispatch(Number, Number)	#snap
def floor(value, step): return math.floor(value / step) * step #returns the greatest multiple of step less than or equal to value

@dispatch(Number, Number)
def ceil(value, step): return math.ceil(value / step) * step #returns the smallest multiple of step greater than or equal to value

def mean(iterable): return sum(iterable) / len(iterable)

"""@dispatch(Number, Number, Number)
def clamp(value, min, max):
	if value < min:
		return min
	elif value > max:
		return max
	else:
		return value

@dispatch(tuple, tuple, float)
def clamp(value, min, max):
	return tuple(map(lambda x, y, z: clamp(x, y, z), value, min, max))

def clamp01(value): return clamp(value, 0.0, 1.0)

def repeat(t, length):
	return clamp(t - math.floor(t / length) * length, 0.0, length)

def ping_pong(t, length):
	t = repeat(t, length * 2)
	return length - abs(t - length)


@dispatch(Number, Number, float)
def lerp(a, b, t):
	return a * (1 - t) + b * t	#a + (b - a) * t

@dispatch(int, int, float)
def lerp(a, b, t):
	return int(a * (1 - t) + b * t)

@dispatch(list, list, float)
def lerp(a, b, t):
	return list(map(lambda x, y: lerp(x, y, t), a, b))

@dispatch(tuple, tuple, float)
def lerp(a, b, t):
	return tuple(map(lambda x, y: lerp(x, y, t), a, b))

@dispatch(list, list, float)
@dispatch(tuple, tuple, float)
def inverse_lerp(a, b, c): return distance(a, c) / distance(a, b) #NOT COMPLETE YET

@dispatch(Number, Number, Number)
def inverse_lerp(a, b, c): return (c - a) / (b - a)

def sqr_magnitude(v): return reduce(lambda a, b: a+b*b, v)
def magnitude(v): return math.sqrt(sqr_magnitude(v))
def sub(a, b): return map(lambda x, y: x - y, a, b)
def raised_norm(a, n): return sum(abs(x) ** n for x in a)
def norm(a, n): return raised_norm(a, n) ** (1.0 / n)
def raised_distance(a, b, n=2): return raised_norm(list(map(lambda x, y: y - x, a, b)), n)
def distance(a, b, n=2): return raised_distance(a, b, n) ** (1.0 / n)
#def distance(a, b): return magnitude(sub(a, b))








@dispatch(Number, Number)
def random_range(start, stop):
	return random.uniform(start, stop)

@dispatch(int, int)
def random_range(start, stop):
	return random.randrange(start, stop + 1)

@dispatch(tuple, tuple)#, float)
def random_range(start, stop):
	return tuple(map(lambda x, y: random_range(x, y), start, stop))"""

def rangef(start, stop, step):
	return [start + step * i for i in range(math.ceil((stop - start) / step))]

def slope(line):
	delta = line[1] - line[0]
	return delta[1] / delta[0] if delta[0] != 0 else float('inf')  # Avoid division by zero


def line_line_intersection(line1, line2):
	(x1, y1), (x2, y2) = line1
	(x3, y3), (x4, y4) = line2

	# Coeficientes da equação geral da reta: Ax + By + C = 0
	A1 = y2 - y1
	B1 = x1 - x2
	C1 = x2 * y1 - x1 * y2

	A2 = y4 - y3
	B2 = x3 - x4
	C2 = x4 * y3 - x3 * y4

	# Determinante
	det = A1 * B2 - A2 * B1

	if np.isclose(det, 0):
		return None  # Retas paralelas ou coincidentes

	# Fórmulas de Cramer
	x = (B1 * C2 - B2 * C1) / det
	y = (A2 * C1 - A1 * C2) / det

	return np.array([x, y])

"""def colinear_point_on_ray(ray, pt):
	return np.dot(ray[1] - ray[0], pt - ray[0]) >= 0	# Checa se projeta no mesmo sentido"""

def point_on_ray(ray, pt, tol=1e-6):	#ray = (origin, direction)
	dir = npx.normalize(ray[1])
	v = pt - ray[0]
	return np.allclose(npx.normalize(v), dir, atol=tol) and np.dot(v, dir) >= -tol
	"""d = ray[1] - ray[0]
	v = pt - ray[0]
	dist = np.linalg.norm(v - np.dot(v, d) / np.dot(d, d) * d)	# Distance from point to ray line
	return dist < tol and np.dot(d, v) >= -tol"""

def colinear_point_on_segment(seg, pt):	# Checa se ponto está no segmento (entre A e B)
	AB = seg[1] - seg[0]
	AP = pt - seg[0]
	dot1 = np.dot(AB, AP)
	dot2 = np.dot(AB, AB)
	return 0 <= dot1 <= dot2
	
def ray_line_intersection(ray, line):
	pt = line_line_intersection(ray, line)	# Calcula interseção
	if pt is None:
		return None
	return pt if colinear_point_on_ray(ray, pt) else None

def segment_line_intersection(seg, line):
	pt = line_line_intersection(seg, line)	# Calcula interseção
	if pt is None:
		return None
	return pt if colinear_point_on_segment(seg, pt) else None

def ray_ray_intersection(ray1, ray2):
	pt = line_line_intersection(ray1, ray2)	# Calcula interseção
	if pt is None:
		return None
	return pt if colinear_point_on_ray(ray1, pt) and colinear_point_on_ray(ray2, pt) else None

def segment_ray_intersection(seg, ray):
	pt = line_line_intersection(seg, ray)	# Calcula interseção
	if pt is None:
		return None
	return pt if colinear_point_on_ray(ray, pt) and colinear_point_on_segment(seg, pt) else None

def segment_segment_intersection(seg1, seg2):
	pt = line_line_intersection(seg1, seg2)	# Calcula interseção
	if pt is None:
		return None
	return pt if colinear_point_on_segment(seg1, pt) and colinear_point_on_segment(seg2, pt) else None

def segment_polygon_intersection(seg, polygon):
	p1, p2 = np.array(seg[0], float), np.array(seg[1], float)
	segment_dir = p2 - p1
	points = []

	# Go through all polygon edges
	for i in range(len(polygon)):
		a = np.array(polygon[i], float)
		b = np.array(polygon[(i + 1) % len(polygon)], float)

		# Edge direction
		edge = b - a
		den = segment_dir[0] * edge[1] - segment_dir[1] * edge[0]
		if abs(den) < 1e-9:
			continue  # parallel → no intersection

		diff = a - p1
		t = (diff[0] * edge[1] - diff[1] * edge[0]) / den
		u = (diff[0] * segment_dir[1] - diff[1] * segment_dir[0]) / den

		if 0 <= t <= 1 and 0 <= u <= 1:
			# Intersection point
			points.append(p1 + t * segment_dir)

	if geo.polyline.contains_point(polygon, p1):
		points.append(p1)
	if geo.polyline.contains_point(polygon, p2):
		points.append(p2)

	if len(points) < 2:
		return None

	# Sort points along the segment direction
	points = sorted(points, key=lambda p: np.dot(p - p1, segment_dir))
	return [points[0], points[-1]]

def segment_rect_intersection(seg, rect): return segment_polygon_intersection(seg, npx.rect2.corners(rect))

"""def line_segment_intersection(line, segment):
	point = line_line_intersection(line, segment)
	if point == None: return None
	t = inverse_lerp(segment[0], segment[1], point)
	if t < 0 or t > 1: return None
	return point"""

"""def random_in_circle(radius, center):
	vector = polar_to_cartesian(random_range(0.0, radius), random_range(0.0, 2.0 * math.pi))
	return tuple(map(lambda x, y: x + y, center, vector))"""

def project_point_on_line(p, a, b):	#Project point p onto the line defined by points a and b.
	return a + npx.project(p - a, b - a)

def project_circle_on_line(center, radius, a, b):
	dir = npx.normalize(b - a)
	proj_center = project_point_on_line(center, a, b)
	return geo.line([proj_center - dir * radius, proj_center + dir * radius])

def circle_circle_distance(center, radius, circle):
	dir_vec = circle.center - center
	dist_centers = np.linalg.norm(dir_vec)
	if dist_centers == 0:
		closest_point = circle.center
	else:
		closest_point = circle.center - dir_vec / dist_centers * circle.radius
	return dist_centers - (circle.radius + radius), closest_point

def point_line_distance(p, a, b):
	proj = project_point_on_line(p, a, b)
	return np.linalg.norm(proj - p), proj

def point_segment_distance(p, a, b):
	#print(a, b)
	t = npx.inverse_lerp(a, b, p)
	closest_point = npx.lerp(a, b, np.clip(t, 0, 1))
	return np.linalg.norm(closest_point - p), closest_point

def circle_line_distance(center, radius, a, b): return point_line_distance(center, a, b) - radius
"""Subtracting the circle’s radius gives the signed distance from the circle’s edge to the line:
	If the result is positive, the line is outside the circle.
	If the result is zero, the line is tangent to the circle.
	If the result is negative, the line intersects the circle."""

def circle_segment_distance(center, radius, a, b):
	dist = point_segment_distance(center, a, b)
	return dist[0] - radius, dist[1]

def circle_polyline_distance(center, radius, vertices):
	dists = [circle_segment_distance(center, radius, *x) for x in vertices.edges()]
	dists.sort(key=lambda x: x[0])
	return dists[0]
	#return min([circle_segment_distance(center, radius, *x) for x in vertices.edges()])

def circle_rect_distance(center, radius, rect):
	return circle_polyline_distance(center, radius, npx.rect2.corners(rect))


def batch(total, size): return [min(total - (i * size), size) for i in range(math.ceil(total / size))]


#{0 1 2}
#0 1 2 10 11 12 20 21 22 100 ... #Base-3 Numeral System
#0 1 2 00 01 02 10 11 12 20 21 22 000 ... #Alphabetic Enumeration (Spreadsheet-like)

def to_number(s, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
	try:
		return sum(index_of(alphabet, x) * (26 ** i) for i, x in enumerate(reversed(s)))
	except:
		print(f'ValueError: invalid literal: {s}')
