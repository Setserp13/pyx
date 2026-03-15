import numpy as np
import pyx.numpyx as npx
import pyx.mat.mat as mat
from pyx.collectionsx import lshift
import math
import pyx.osx as osx
from pyx.collectionsx import List
from pyx.collectionsx import flatten
from itertools import product

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
	def area(self): return 2 * math.pi * self.radius ** 2

	@property
	def aabb(self): return npx.rect.center_size(self.center, np.ones(2) * self.diameter)
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


class ellipse():
	def __init__(self, center, a, b):	#a and be are semi axes
		self.center = np.array(center)
		self.a = a
		self.b = b

	def get_point(self, theta): return np.array([self.a * math.cos(theta), self.b * math.sin(theta)]) + self.center

	@property
	def aabb(self): return npx.rect.center_size(self.center, np.array([self.a, self.b]) * 2)
	@aabb.setter
	def aabb(self, value):
		self.center = value.center
		self.a = value.extents[0]
		self.b = value.extents[1]

	def __rmatmul__(self, M): return self.__matmul__(M)

	def __matmul__(self, M):
		M = np.asarray(M, dtype=float)

		if M.shape != (3, 3):
			raise ValueError("Expected a 3x3 affine matrix")

		# Transform center
		center_h = np.append(self.center, 1)
		center = (M @ center_h)[:2]

		# Linear part
		A = M[:2, :2]

		# Transform axis vectors
		a_vec = A @ np.array([self.a, 0])
		b_vec = A @ np.array([0, self.b])

		a = np.linalg.norm(a_vec)
		b = np.linalg.norm(b_vec)

		return ellipse(center, a, b)
		
	def copy(self): return ellipse(self.center.copy(), self.a, self.b)

class arc(circle):
	def __init__(self, center, radius, start, end): #start is start angle and end is end angle
		super().__init__(center, radius)
		self.start = start
		self.end = end

	def get_point01(self, t):
		t = npx.clamp01(t)
		return self.get_point(npx.lerp(self.start, self.end, t))

class line(np.ndarray):	#start = self[0], end = self[1]
	def __new__(cls, input_array):
		# Convert input to ndarray and view it as MyArray
		obj = np.asarray(input_array).view(cls)
		return obj

	def __array_finalize__(self, obj):
		# Called when the object is created via view/slicing
		if obj is None: return
		# You can set custom attributes here if needed
		self.my_attribute = getattr(obj, 'my_attribute', 'default')

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

	"""def normal(self, left=True):
		v = self.vector
		n = np.array([-v[1], v[0]]) if left else np.array([v[1], -v[0]])
		return n if np.array_equal(self[0], self[1]) else npx.normalize(n)"""

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
	
	def subdivide(self, n): return polyline(npx.subdivide(self[0], self[1], n+1), closed=False).edges()

	@property
	def aabb(self): return npx.aabb(*self)
	@aabb.setter
	def aabb(self, value): self[:] = npx.set_aabb(self, value)

	def __rmatmul__(self, M): return self.__matmul__(M)
	def __matmul__(self, M): return line(npx.affine_transform(M, self))

	def set_position(self, pivot, value):	#pivot is normalized and value is not normalized
		pivot = npx.lerp(*self, pivot)
		delta = value - pivot
		return line(self + delta)


def point_on_line(line, point, tol=1e-8): #tol: tolerância numérica
	line_vec = line[1] - line[0]
	test_vec = point - line[0]
	cross = np.cross(line_vec, test_vec)
	return np.abs(cross) < tol

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
	for x in vertices.edges(closed=vertices.closed):	#polyline.edges(vertices, closed=closed):
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
	def __init__(self, vertices=None, faces=None, uvs=None):
		self.vertices = vertices.copy() if vertices is not None else []
		self.faces = faces.copy() if faces is not None else []
		self.uvs = uvs.copy() if uvs is not None else []
		self.normals_interpolation = 'face_varying'	#constant (one value for entire primitive), face_varying (one value per face corner), vertex (one value per vertex), uniform (one value per face)
		self.uvs_interpolation = 'face_varying'
		self.colors = None
		self.colors_interpolation = 'face_varying'
		self.double_sided = False
	
	def get_face(self, i): return polyline(List.items(self.vertices, self.faces[i]))

	def get_faces(self): return group([self.get_face(i) for i in range(len(self.faces))])

	@property
	def edges(self): return flatten([polyline.edges(x) for x in self.get_faces()])
	
	def to_obj(self, path):
		lines = [f"v {x} {y} {z}" for x, y, z in self.vertices]
		if self.uvs is not None and len(self.uvs) > 0:
			lines += [f"vt {u} {v}" for u, v in self.uvs]
			lines += ["f " + " ".join(f"{i+1}/{i+1}" for i in face) for face in self.faces]
		else:
			lines += [f"f {' '.join(str(i + 1) for i in face)}" for face in self.faces]
		osx.write(path, '\n'.join(lines) + '\n')

	def from_obj(path):
		vertices = []
		uvs = []
		faces = []
		for line in osx.read(path).split('\n'):
			if line.startswith('v '):
				_, x, y, z = line.strip().split()
				vertices.append([float(x), float(y), float(z)])
			elif line.startswith('vt '):
				_, u, v = line.strip().split()
				uvs.append([float(u), float(v)])
			elif line.startswith('f '):
				parts = line.strip().split()[1:]
				face = []
				for p in parts:
					tokens = p.split('/')
					vertex_index = int(tokens[0]) - 1
					face.append(vertex_index)
					# Optional: If you want to store UV indices separately, you can do:
					# if len(tokens) > 1 and tokens[1]:
					#     uv_index = int(tokens[1]) - 1
					#     self.face_uv_indices.append(uv_index)
				faces.append(face)
		return Mesh(vertices, faces, uvs)
	
	def add_face(self, vertices):
		start_index = len(self.vertices)
		self.vertices.extend(vertices)
		self.faces.append(list(range(start_index, start_index + len(vertices))))

	def add_faces(self, faces):
		for x in faces:
			self.add_face(x)

	def extrude(self, dir, face, flip=True):
		face = self.get_face(face)
		self.add_face([x + dir for x in face])
		for x in polyline.edges(face):
			self.add_face([x[0], x[1], x[1] + dir, x[0] + dir])
		if flip:
			for i in range(len(self.faces) - 5, len(self.faces)):
				self.flip_normal(i)

	def translate(self, vector):
		self.vertices = [x + vector for x in self.vertices]

	def scale(self, vector):
		self.vertices = [x * vector for x in self.vertices]

	@property
	def aabb(self): return npx.aabb(self.vertices)	#bounds

	@property
	def pivot(self): return self.bounds.normalize_point(np.zeros(3))

	@pivot.setter	#pivot is normalized
	def pivot(self, value): self.translate(-self.bounds.denormalize_point(value))

	def merge(*args):
		vertices = []
		faces = []
		for x in args:
			faces += [[len(vertices) + z for z in y] for y in x.faces]
			vertices += x.vertices
		return Mesh(vertices, faces)

	def make_double_sided(mesh):
		mesh.faces += [list(reversed(x)) for x in mesh.faces]

	def flip_normal(mesh, i):
		mesh.faces[i] = list(reversed(mesh.faces[i]))

	def flip_normals(mesh):
		mesh.faces = [list(reversed(x)) for x in mesh.faces]

	def make_vertices_unique(self):
		result = Mesh()
		for x in self.get_faces():
			result.add_face(*x)
		return result

	def face_normal(self, i):
		pts = self.get_face(i)
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
		return [self.face_normal(i) for i in range(len(self.faces))]

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

		for fi, face in enumerate(self.faces):
			for vi in face:
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
			for fi, face in enumerate(self.faces)
			for _ in face
		])

	@property
	def smooth_corner_normals(self):	#face varying smooth
		vnormals = self.vertex_normals
		return np.asarray([
			vnormals[vi]
			for face in self.faces
			for vi in face
		])


class angle(list):
	def rays(self): return [line([self[1], self[0]]), line([self[1], self[2]])]

	def vectors(self): return [x.vector for x in self.rays()]

	def size(self): return npx.angle(*self.vectors())

EPSILON = 1e-10

class polyline(np.ndarray):#list):


	def __new__(cls, input_array, closed=True):
		# Convert input to ndarray and view it as MyArray
		obj = np.asarray(input_array).view(cls)
		obj.closed = closed
		return obj

	def __array_finalize__(self, obj):
		# Called when the object is created via view/slicing
		if obj is None: return
		# You can set custom attributes here if needed
		self.my_attribute = getattr(obj, 'my_attribute', 'default')

	def edges(p): return [line(x) for x in List.aranges(p, 2, cycle=p.closed)]
	
	def edge(p, index): return line([p[index], p[(index + 1) % len(p)]])
	
	def vertex_angle(p, index): return angle(List.arange(p, 3, start=index - 1))
	
	def vertex_angles(p): return [p.vertex_angle(i) for i in range(0 if p.closed else 1, len(p) - (0 if p.closed else 1))]

	def lengths(p): return [x.length for x in p.edges()]
	
	def perimeter(p): return sum(p.lengths())
	
	def midpoints(p): return [x.midpoint for x in p.edges()]
	
	def point_from_proportion(self, t):
		p = self.perimeter()
		a = 0.0
		for x in self.edges():
			b = a + x.length / p
			if a <= t and t <= b:
				return npx.lerp(x[0], x[1], (t - a) / (b - a))
			a = b
		return None

	def subdivide(p, n):
		result = []
		for x in p.edges():
			result += [npx.lerp(x[0], x[1], i / n) for i in range(n)]
		if not p.closed:
			result.append(vertices[-1])
		return polyline(result, closed=p.closed)

	def incident_edges(p, vertex): #vertex is an index
		edges = [List.arange(p, 2, start=vertex - 1), List.arange(p, 2, start=vertex)]
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
		return result

	def normal(edge, outward=True): return line(edge).normal * -1 if outward else 1

	def normals(p, outward=True): return [polyline.normal(x, outward=outward) for x in p.edges()]
	
	def vertex_normals(vertices, outward=True):
		return [npx.normalize(np.sum([polyline.normal(x, outward=outward) for x in p.incident_edges(i)], axis=0)) for i in range(len(p))]

	def expand(self, amount, outward=True):
		return polyline([self[i] + x * float(amount) for i, x in enumerate(self.vertex_normals(outward=outward, closed=self.closed))], closed=self.closed)
	
	def internal_angle_sum(n): return math.pi * (n - 2)

	def perpendicular_bisector(edge):
		mid = np.mean(edge, axis=0)
		return [mid, mid + polyline.normal(edge, outward=False)]

	def perpendicular_bisectors(p): return [polyline.perpendicular_bisector(x) for x in p.edges()]

	def circumcenter(p): return mat.line_line_intersection(*p.perpendicular_bisectors()[:2])

	def centroid(p): return np.mean(p, axis = 0)

	def line_intersection(p, line): return [mat.segment_line_intersection(x, line) for x in p.edges()]

	def ray_intersection(p, ray): return [mat.segment_ray_intersection(x, ray) for x in p.edges()]

	def segment_intersection(p, seg): return [mat.segment_segment_intersection(x, seg) for x in p.edges()]

	def is_clockwise(polygon):
		sum = 0
		for i in range(len(polygon)):
			x1, y1 = polygon[i]
			x2, y2 = polygon[(i + 1) % len(polygon)]
			sum += (x2 - x1) * (y2 + y1)
		return sum > 0

	def area(vertices):
		area = 0.0
		n = len(vertices)
		for i in range(n):
			x1, y1 = vertices[i]
			x2, y2 = vertices[(i + 1) % n]
			area += (x1 * y2 - x2 * y1)
		return area * 0.5	#abs(area) * 0.5, let it return negative numbers too, to triangulate works fine

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
	
	def triangulate(vertices):
		n = len(vertices)
		if n < 3:
			return []
	
		indices = list(range(n))
		if polyline.is_clockwise(vertices):
			indices.reverse()
	
		triangles = []
	
		while len(indices) > 3:
			found_ear = False
			for i in range(len(indices)):
				i0 = indices[i]
				i1 = indices[(i + 1) % len(indices)]
				i2 = indices[(i + 2) % len(indices)]
	
				tri = [vertices[i0], vertices[i1], vertices[i2]]
				if polyline.area(tri) <= 0:
					continue
	
				ear_found = True
				for j in indices:
					if j in (i0, i1, i2):
						continue
					if polyline.contains_point(tri, vertices[j]):
						ear_found = False
						break
	
				if ear_found:
					triangles.append([i0, i1, i2])
					del indices[(i + 1) % len(indices)]
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
			for x in v.edges():
				normal = polyline.normal(x, outward=True)
				result.append(x[0] + normal * width * (1.0 - align))
				result.append(x[1] + normal * width * (1.0 - align))
				result.insert(0, x[0] - normal * width * align)
				result.insert(0, x[1] - normal * width * align)
		elif join == 'butt':
			for i, x in enumerate(v.vertex_normals(outward=True)):
				result.append(v[i] + x * width * (1.0 - align))
				result.insert(0, v[i] - x * width * align)
		elif join == 'miter':
			angles = [x.size() for x in polyline.vertex_angles(v, closed=closed)]
			if not v.closed:
				angles = [math.pi] + angles + [math.pi]
			for i, x in enumerate(v.vertex_normals(outward=True)):
				s = math.sin(angles[i] / 2)
				result.append(v[i] + x * (width * (1.0 - align)) / s)
				result.insert(0, v[i] - (x * width * align) / s)
		if v.closed:
				mid = len(result) // 2
				#print(result, mid, result[mid - 1])
				return polyline([result[mid - 1]] + result + [result[mid]])
		else: return polyline(result)

	def rotate_around(p, angle, center=np.zeros(2)): return polyline([npx.rotate_around(x, angle, center) for x in p], closed=p.closed)

	def from_vectors(v, axis=None, start=np.zeros(2)):	#concatenation of n vectors (or edges) end-to-end starting from start
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
		return result

	@property
	def aabb(self): return npx.aabb(self)
	@aabb.setter
	def aabb(self, value): self[:] = npx.set_aabb(self, value)

	def __rmatmul__(self, M): return self.__matmul__(M)
	def __matmul__(self, M): return polyline(npx.affine_transform(M, self), closed=self.closed)

	def clip(p1, p2):	#split the edges of polyline p1 wherever they intersect edges of polyline p2
		result = []
		for edge in p1.edges():
			result.append(edge[0])
			inter = [
				p for y in p2.edges()
				if (p := mat.segment_segment_intersection(edge, y)) is not None
				and not any(np.array_equal(p, v) for v in edge)
			]
			if inter:
				result.extend(sort_by_distance(inter, edge[0]))
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
	def angle_bisectors(vertices):
		result = []
		for i in range(3):
			A, B, C = List.arange(vertices, 3, i)
			b = np.linalg.norm(C - A)
			c = np.linalg.norm(A - B)
			P = (B * b + C * c) / (b + c)
			result.append([A, P])
		return result
	
	def altitudes(vertices):
		result = []
		for i in range(3):
			P, A, B = List.arange(vertices, 3, i)
			AB = B - A
			t = np.dot(P - A, AB) / np.dot(AB, AB)
			foot = A + t * AB
			result.append([P, foot])
		return result

	def medians(vertices):
		return list(zip(vertices, lshift(polyline.midpoints(vertices, closed=True))))
	
	def incenter(vertices):
		A, B, C = vertices
		c, a, b = polyline.lengths(vertices, closed=True)
		P = a + b + c
		return (a*A + b*B + c*C) / P
	


	def orthocenter(vertices):
		A, B, C = vertices
		def altitude_line(p_vertex, p1, p2):
			edge_vec = p2 - p1
			perp_vec = np.array([-edge_vec[1], edge_vec[0]])
			return p_vertex, perp_vec
	
		h1, d1 = altitude_line(A, B, C)
		h2, d2 = altitude_line(B, A, C)
	
		A_mat = np.array([d1, -d2]).T
		b_vec = h2 - h1
		t = np.linalg.solve(A_mat, b_vec)
		return h1 + t[0] * d1




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
	faces = []
	for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
		faces += prism_laterals(count, start_index1=count * a, start_index2=count * b)
	return vertices, faces

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
		return npx.aabb([x.aabb for x in self])

	@aabb.setter
	def aabb(self, value):
		cur = self.aabb
		for x in self:
			x.aabb = value.denormalize_rect(cur.normalize_rect(x.aabb))

def distribute(arr, axis=0, align=.5, gap=0.0):
	for i in range(1, len(arr)):
		pos = arr[i - 1].aabb.denormalize_point(np.array([1, float(align)])[[axis, 1 - axis]]) + npx.ei(axis, 2) * float(gap)
		arr[i].aabb = arr[i].aabb.set_position(pivot = np.array([0, float(align)])[[axis, 1 - axis]], value = pos)

def rects(offset, sizes, axis=0, align=0.5, gap=0.0):
	dim = len(sizes[0])
	result = group([npx.rect(np.zeros(dim), x) for x in sizes])
	result[0].min = offset
	#print(offset)
	distribute(result, axis=axis, align=align, gap=gap)
	return result

def collinear(u, v, tol=1e-12):	# check if vectors u and v are collinear
	#return abs(np.cross(u, v)) < tol	# works in 2D only
	return np.linalg.norm(np.cross(u, v)) < tol	# works in 2D and 3D

def point_on_segment(seg, p, tol=1e-12):	# check if point p is on segment ab
	a, b = seg
	ap = p - a
	ab = b - a
	if not collinear(ab, ap, tol):
		return False
	return -tol <= np.dot(ap, ab) <= np.dot(ab, ab) + tol

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





















