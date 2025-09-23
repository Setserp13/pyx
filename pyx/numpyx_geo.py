import numpy as np
import pyx.numpyx as npx
import pyx.mat.mat as mat
from pyx.collectionsx import lshift
import math
import pyx.osx as osx
from pyx.collectionsx import List
from itertools import product

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

"""class line():
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
	def normal(self): return np.array([-self.direction[1], self.direction[0]] + list(self.direction[2:]))"""

class line(np.ndarray):
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

	@property #in XY-plane
	def normal(self): return np.array([-self.direction[1], self.direction[0]] + list(self.direction[2:]))

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


def truncate(vertices, t=.25, closed=True):
	result = []
	for x in polyline.edges(vertices, closed=closed):
		result += [npx.lerp(x[0], x[1], t), npx.lerp(x[0], x[1], 1-t)]
	if not closed:
		result[0] = vertices[0]
		result[-1] = vertices[-1]
	return result

#returns the vertex indices of a polygon, a star polygon or a polygon compound, is denoted by its Schläfli symbol {p/q}, where p and q are relatively prime (they share no factors) and q ≥ 2
def polygram(p, q):	#p = total number of vertices, q = step size (how many points to skip when drawing)
	if q >= p / 2 or p < 3:
		return []
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

	def get_face(self, i): return List.items(self.vertices, self.faces[i])

	def get_faces(self): return [self.get_face(i) for i in range(len(self.faces))]

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
	
	def add_face(self, *vertices):
		start_index = len(self.vertices)
		self.vertices.extend(vertices)
		self.faces.append(list(range(start_index, start_index + len(vertices))))

	def extrude(self, dir, face):
		face = self.get_face(face)
		self.add_face(*[x + dir for x in face])
		for x in polyline.edges(face):
			self.add_face(x[0], x[1], x[1] + dir, x[0] + dir)

	def translate(self, vector):
		self.vertices = [x + vector for x in self.vertices]

	def scale(self, vector):
		self.vertices = [x * vector for x in self.vertices]

	@property
	def bounds(self): return npx.aabb(self.vertices)

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

class angle(list):
	def rays(self): return [line([self[1], self[2]]), line([self[1], self[0]])]

	def vectors(self): return [x.vector for x in self.rays()]

	def size(self): return npx.angle(*self.vectors())

EPSILON = 1e-10

class polyline(np.ndarray):#list):


	def __new__(cls, input_array):
		# Convert input to ndarray and view it as MyArray
		obj = np.asarray(input_array).view(cls)
		return obj

	def __array_finalize__(self, obj):
		# Called when the object is created via view/slicing
		if obj is None: return
		# You can set custom attributes here if needed
		self.my_attribute = getattr(obj, 'my_attribute', 'default')


	
	def edges(p, closed=True): return List.aranges(p, 2, cycle=closed)

	def edge(vertices, index): return [vertices[index], vertices[(index + 1) % len(vertices)]]
	
	#def angle(vertices, index): return [vertices.edge(index), list(reversed(vertices.edge((index - 1) % len(vertices))))]
	def vertex_angle(vertices, index): return angle(List.arange(vertices, 3, start=index - 1))
	
	def vertex_angles(vertices): return [vertices.vertex_angle(i) for i in range(len(vertices))]
	
	"""def angle_size(vertices, index):
		angle = vertices.angle(index)
		return npx.angle(angle[0][1] - angle[0][0], angle[1][1] - angle[1][0])"""
	
	def lengths(vertices, closed=True):
		return [np.linalg.norm(x[0] - x[1]) for x in polyline.edges(vertices, closed=closed)]
	
	def perimeter(vertices, closed=True): return sum(lengths(vertices, closed=closed))

	def midpoints(vertices, closed=True):
		return [np.mean(x, axis = 0) for x in polyline.edges(vertices, closed=closed)]
	
	def point_from_proportion(t, vertices, closed=True):
		p = polyline.perimeter(vertices, closed=closed)
		a = 0.0
		for x in polyline.edges(vertices, closed=closed):
			b = a + np.linalg.norm(x[0] - x[1]) / p
			if a <= t and t <= b:
				return npx.lerp(x[0], x[1], (t - a) / (b - a))
			a = b
		return None

	def subdivide(vertices, n, closed=True):
		result = []
		for x in polyline.edges(vertices, closed=closed):
			result += [npx.lerp(x[0], x[1], i / n) for i in range(n)]
		if not closed:
			result.append(vertices[-1])
		return result

	def incident_edges(vertices, vertex, closed=True): #vertex is an index
		edges = [List.arange(vertices, 2, start=vertex - 1), List.arange(vertices, 2, start=vertex)]
		if not closed:
			if vertex == 0:
				return edges[1:]
			elif vertex == len(vertices) - 1:
				return edges[:-1]
		return edges

	def normal(edge, outward=True):
		v = edge[1] - edge[0]
		result = np.array([v[1], -v[0]]) if outward else np.array([-v[1], v[0]])
		return npx.normalize(result)
	
	def normals(vertices, outward=True, closed=True): return [polyline.normal(x, outward=outward) for x in polyline.edges(vertices, closed)]
	
	def vertex_normals(vertices, outward=True, closed=True):
		return [npx.normalize(np.sum([polyline.normal(x, outward=outward) for x in polyline.incident_edges(vertices, i, closed=closed)], axis=0)) for i in range(len(vertices))]

	def internal_angle_sum(n): return math.pi * (n - 2)

	def perpendicular_bisector(edge):
		mid = np.mean(edge, axis=0)
		return [mid, mid + polyline.normal(edge, outward=False)]

	def perpendicular_bisectors(vertices, closed=True):
		return [polyline.perpendicular_bisector(x) for x in polyline.edges(vertices, closed=closed)]

	def circumcenter(vertices, closed=True):
		return mat.line_line_intersection(*polyline.perpendicular_bisectors(vertices, closed=closed)[:2])
		"""m1, v1 = polyline.perpendicular_bisector(vertices[:2])
		m2, v2 = polyline.perpendicular_bisector(vertices[1:3])
	
		A_mat = np.array([v1, -v2]).T
		b_vec = m2 - m1
		t = np.linalg.solve(A_mat, b_vec)
		return m1 + t[0] * v1"""

	def centroid(vertices): return np.mean(vertices, axis = 0)

	def line_intersection(vertices, line, closed=True): return [mat.segment_line_intersection(x, line) for x in polyline.edges(vertices, closed=closed)]

	def ray_intersection(vertices, ray, closed=True): return [mat.segment_ray_intersection(x, ray) for x in polyline.edges(vertices, closed=closed)]

	def segment_intersection(vertices, seg, closed=True): return [mat.segment_segment_intersection(x, seg) for x in polyline.edges(vertices, closed=closed)]

	def contains_point(vertices, point):
		return len(polyline.ray_intersection(vertices, [np.centroid(vertices), vertices[0]], closed=True)) % 2 == 1

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


	def to_stroke(v, width=1, closed=False):
		inward_normals = polyline.vertex_normals(v, outward=False, closed=closed)
		outward_normals = polyline.vertex_normals(v, outward=True, closed=closed)
		for i, x in enumerate(v):
			inward_normals[i] = inward_normals[i] * width + x
			outward_normals[i] = outward_normals[i] * width + x
		return polyline(inward_normals + list(reversed(outward_normals)))




class polygon:
	def a(n, R=1): return R * math.cos(math.pi/n)	#apothem

	def R(n, a=1): return a / math.cos(math.pi/n)

	def s(n, R=1): return 2 * R * math.sin(math.pi/n)

	def internal_angle(n): return polyline.internal_angle_sum(n) / n


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



class bezier:
	def get_point(t, *p):
		# Recursive Bézier evaluation (De Casteljau's algorithm)
		if len(p) == 1:
			return p[0]
		else:
			return (1 - t) * get_point(t, *p[:-1]) + t * get_point(t, *p[1:])

		"""#Evaluate a Bézier curve at parameter t using Bernstein basis.
		n = len(points) - 1
		points = np.array(points)
		point = np.zeros_like(points[0])
		for i in range(n + 1):
			binom = np.math.comb(n, i)
			point += binom * ((1 - t) ** (n - i)) * (t ** i) * points[i]
		return point"""
	
	def get_derivative(t, *p):
		# Derivative of Bézier curve (based on differences between control points)
		n = len(p) - 1
		if n < 1:
			raise ValueError("Need at least two points for a derivative")
		derivative_points = [n * (p[i + 1] - p[i]) for i in range(n)]
		return get_point(t, *derivative_points)


def randomize2(vertices, r=.1): return [npx.random_in_circle(r) + x for x in vertices]

def randomize3(vertices, r=.1): return [npx.random_in_sphere(r) + x for x in vertices]


def bars(values, offset = np.zeros(2), width=1, gap=0, axis=0, align=0):
	result = []
	for i, y in enumerate(values):
		min = offset + np.array([i * (width + gap), -y * align])[[axis, 1 - axis]] #swizzle
		size = np.array([width, y])[[axis, 1 - axis]]
		result.append(npx.rect(min, size))
	return result










