import numpy as np
import pyx.numpyx as npx
import pyx.mat.mat as mat
from pyx.collectionsx import lshift
import math
import pyx.osx as osx

class circle():
	def __init__(self, center, radius):
		self.center = np.array(center)
		self.radius = radius

	def angle_of(self, point):
		delta = point - self.center
		return np.arctan2(delta[1], delta[0])

	def get_point(self, theta): return np.array(list(mat.polar_to_cartesian(self.radius, theta)) + [0]) + self.center		

	def get_point01(self, t): return self.get_point(t * 2 * np.pi)

class arc(circle):
	def __init__(self, center, radius, start, end): #start is start angle and end is end angle
		super().__init__(center, radius)
		self.start = start
		self.end = end

	def get_point01(self, t):
		t = mat.clamp01(t)
		return self.get_point(mat.lerp(self.start, self.end, t))

class line():
	def __init__(self, start, end):
		self.start = np.array(start)
		self.end = np.array(end)

	@property
	def length(self): return np.linalg.norm(self.end - self.start)

	@property
	def direction(self): return npx.normalize(self.end - self.start)

	@property
	def midpoint(self): return (self.start + self.end) * 0.5

	@property #in XY-plane
	def normal(self): return np.array([-self.direction[1], self.direction[0]] + list(self.direction[2:]))


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

def truncate(vertices, length=.1):
	result = []
	for i in range(len(vertices)):
		next = vertices[(i+1)%len(vertices)]
		dir = npx.normalize(next - vertices[i])
		result += [vertices[i] + dir * length, next - dir * length]
	return result


def corners(rect): return [rect.denormalize_point(x) for x in [np.array([0,0]), np.array([0,1]), np.array([1,1]), np.array([1,0])]]


def regular_star_polygon(m, n, r=1.0, center=np.zeros(2), start=0.0):	#m = total number of vertices on the circle, n = step size (how many points to skip when drawing)
	vertices = npx.on_circle(n=m, r=r, center=center, start=start)
	return [vertices[(i * n) % m] for i in range(m)]

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
	def __init__(self, vertices=None, faces=None):
		self.vertices = vertices.copy() if vertices else []
		self.faces = faces.copy() if faces else []

	def get_face(self, i): return List.items(self.vertices, self.faces[i])

	def get_faces(self): return [self.get_face(i) for i in range(len(self.faces))]

	def to_obj(self, path):
		lines = [f"v {x} {y} {z}" for x, y, z in self.vertices]
		lines += [f"f {' '.join(str(i + 1) for i in face)}" for face in self.faces]
		osx.write(path, '\n'.join(lines) + '\n')

	def from_obj(self, path):
		for line in osx.read(path).split('\n'):
			if line.startswith('v '):
				_, x, y, z = line.strip().split()
				self.vertices.append([float(x), float(y), float(z)])
			elif line.startswith('f '):
				parts = line.strip().split()[1:]
				# Suporta faces tipo: f 1 2 3 ou f 1/1 2/2 3/3
				face = [int(p.split('/')[0]) - 1 for p in parts]
				self.faces.append(face)
		return self

	def add_face(self, *vertices):
		start_index = len(self.vertices)
		self.vertices.extend(vertices)
		self.faces.append(list(range(start_index, start_index + len(vertices))))

	def extrude(self, dir, face):
		face = self.get_face(face)
		self.add_face(*[x + dir for x in face])
		for x in polygon.edges(face):
			self.add_face(x[0], x[1], x[1] + dir, x[0] + dir)

class polygon:
	def edges(p): return [[p[i], p[(i+1)%len(p)]] for i in range(len(p))]

	def s(n, R=1): return 2 * R * math.sin(math.pi/n)


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
