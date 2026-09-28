import math
import numpy as np
import pyx.numpyx as npx
import pyx.numpyx_geo as geo
from pyx.numpyx_geo import Mesh
from pyx.mat.transform import Node3D
import struct
import json
from pyx.collectionsx import flatten#, Map
import copy
import pyx.mat.mat as mat
import pyx.rex as rex

import struct


def triangles(indices):
	result = []
	for x in indices:
		result += fan_triangulate(x)
	return result

"""Buffer
	└── BufferView
		├── Accessor
		├── Accessor
		└── Accessor"""

class Buffer(Node):
	def __init__(self, uri=None):
		super().__init__()
		self.uri = uri

	def append(self, bytes, target=None):
		view = BufferView(bytes, len(self.bytes), target)
		super().append(view)
		return view

	@property
	def bytes(self): return b"".join(x.bytes for x in self.children)

	def gltf(self):
		return {
			"byteLength": len(self.bytes),
			**(
				{"uri": self.uri}
				if self.uri is not None
				else {}
			)
		}

class BufferView(Node):
	ARRAY_BUFFER = 34962	#vertex data (positions, normals, tangents, UVs, colors, joints, weights etc.)
	ELEMENT_ARRAY_BUFFER = 34963	#index data (faces, triangles)	

	def __init__(self, bytes, byte_offset=0, target=None):
		super().__init__()
		self.bytes = bytes
		self.byte_offset = byte_offset
		self.target = target

	def append(self, component_type, type, byte_offset=0, normalized=False):
		accessor = Accessor(component_type, type, byte_offset, normalized)
		super().append(accessor)
		return accessor

	def gltf(self):
		return {
			"buffer": self.parent.siblingIndex,
			"byteOffset": self.byte_offset,
			"byteLength": len(self.bytes),
			**(
				{"target": self.target}
				if self.target is not None
				else {}
			)
		}

class Accessor(Node):
	COMPONENT_SIZE = {
		5120: 1,	# BYTE
		5121: 1,	# UNSIGNED_BYTE
		5122: 2,	# SHORT
		5123: 2,	# UNSIGNED_SHORT
		5125: 4,	# UNSIGNED_INT
		5126: 4		# FLOAT
	}

	TYPE_SIZE = {
		"SCALAR": 1,
		"VEC2": 2,
		"VEC3": 3,
		"VEC4": 4,
		"MAT2": 4,
		"MAT3": 9,
		"MAT4": 16
	}

	def __init__(self, component_type, type, byte_offset=0, normalized=False):
		super().__init__()
		self.component_type = component_type
		self.type = type
		self.byte_offset = byte_offset
		self.normalized = normalized

	@property
	def size(self): return self.COMPONENT_SIZE[self.component_type] * self.TYPE_SIZE[self.type]

	@property
	def count(self): return (len(self.parent.bytes) - self.byte_offset) // self.size

	def gltf(self):
		return {
			"bufferView": self.parent.siblingIndex,
			"byteOffset": self.byte_offset,
			"componentType": self.component_type,
			"count": self.count,
			"type": self.type,
			**(
				{"normalized": True}
				if self.normalized
				else {}
			)
		}



class GLB:
	def __init__(self):
		self.buffers = Node(children=[Buffer()])

		self.resources = {
			'scenes': [{"nodes":[0]}],
			'nodes': [],
			'meshes': [],
			'skins': [],
			'materials': [],
			'textures': [],
			'images': [],
			'samplers': []
		}

		self.scene = None

	@property
	def buffer(self): return self.buffers.children[0]

	def gltf(self):
		json_chunk = {
			"asset": {"version": "2.0"},
			"scene": 0,

			**self.resources,

			"buffers": [],
			"bufferViews": [],
			"accessors": []
		}

		for x in self.buffers.descendants():
			json_chunk[{ 'Buffer': 'buffers', 'BufferView': 'bufferViews', 'Accessor': 'accessors' }[x.__class__.__name__]].append(x.gltf())

		return json_chunk

	def add_buffer(self, uri=None):
		buffer = Buffer(uri)
		self.buffers.append(buffer)

	def pack(self, filename):
		json_chunk = json.dumps(self.gltf(), separators=(",", ":")).encode("utf-8")

		json_chunk += b" " * (-len(json_chunk) % 4)

		bin = self.buffers.children[0].bytes
		bin += b"\x00" * (-len(bin) % 4)

		magic = 0x46546C67
		version = 2

		total_length = (12 + 8 + len(json_chunk) + 8 + len(bin))

		with open(filename, "wb") as f:
			f.write(struct.pack("<III", magic, version, total_length))
			f.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
			f.write(json_chunk)
			f.write(struct.pack("<I4s", len(bin), b"BIN\0"))
			f.write(bin)

		print("Saved", filename)

	def add_node(self, node):
		result = { 'name': f'Node ({node.index})', 'children': [x.index for x in node.children], 'translation': node.position.tolist(), 'rotation': node.rotation.tolist(), 'scale': node.scale.tolist() }
		if hasattr(node, 'attrib'):
			for k in ['mesh', 'skin']:
				if k in node.attrib:
					result[k] = node.attrib[k].index	
		self.resources['nodes'].append(result)

	def add_mesh(self, mesh):
		result = GLBElement(self.buffer, mode=4)
		result.add('attributes/POSITION', mesh.vertices, 5126, 'VEC3').add('indices', triangles(mesh.indices), 5125, 'SCALAR', 34963)

		if len(mesh.uvs) > 0:
			result.add('attributes/TEXCOORD_0', mesh.uvs, 5126, 'VEC2')

		if getattr(mesh, 'joints', None):
			result.add('attributes/JOINTS_0', mesh.joints, 5125, 'VEC4')

		if getattr(mesh, 'weights', None):
			result.add('attributes/WEIGHTS_0', mesh.weights, 5126, 'VEC4')

		self.resources['meshes'].append({ 'primitives': [ result ] })

	def add_skin(self, skin):
		result = GLBElement(self.buffer, joints=[x.index for x in skin.joints])
		self.resources['skins'].append(result.add('inverseBindMatrices', [x.inv_global_TRS.T for x in skin.joints], 5126, 'MAT4'))



def get(data, path):
	for key in path.split("/"):
		data = data[key]

	return data



def set(data, path, value):
	keys = path.split("/")

	for key in keys[:-1]:
		data = data.setdefault(key, {})	#data = data[key]

	data[keys[-1]] = value

GLTF_DTYPES = {
	5121: "<u1",
	5123: "<u2",
	5125: "<u4",
	5126: "<f4"
}

class GLBElement(dict):
	def __init__(self, buffer, **kwargs):
		super().__init__(**kwargs)
		self.buffer = buffer

	def add(self, path, data, component_type, type, target=34962):
		data = np.asarray(data, dtype=GLTF_DTYPES[component_type])
		self.buffer.append(data.tobytes(), target).append(component_type, type)
		set(self, path, len(self.buffer.children) - 1)
		return self



def to_glb(root):

	result = { 'node': [], 'mesh': [], 'skin': [] }
	
	for x in root.descendants():
		if not x in result['node']:
			x.index = len(result['node'])
			result['node'].append(x)
			if hasattr(x, 'attrib'):
				for k in ['mesh', 'skin']:
					if k in x.attrib:
						y = x.attrib[k]
						if not y in result[k]:
							y.index = len(result[k])
							result[k].append(y)

	glb = GLB()

	for k in result:
		for x in result[k]:
			getattr(glb, f'add_{k}')(x)

	return glb



class Skin():
	def __init__(self, joints=None):
		self.joints = [] if joints is None else joints	# list[ list[Node2D or Node3D] ]



def fan_triangulate(indices):
	return [[indices[0], indices[i], indices[i+1]] for i in range(1, len(indices) - 1)]


#MESHING

def uv_sphere(radius=1.0, stacks=16, slices=32, center=np.zeros(3)):
	v = []
	uv = []
	for theta in npx.subdivide(0.0, math.pi, stacks):
		for phi in npx.subdivide(0.0, math.pi * 2.0, slices):
			v.append(npx.spherical_to_cartesian(radius, theta, phi) + center)
			uv.append(np.array([phi / (math.pi * 2), theta / math.pi]))
	return geo.Mesh(v, geo.enlongated_faces(*([slices] * stacks)), uv)

def generate_rings(polyline, ring_radius=1.0, ring_segments=16):
	"""
	polyline: (N,3) numpy array of 3D points
	ring_radius: radius of the cross-sectional ring
	ring_segments: number of points in each ring

	returns: list of numpy arrays shaped (ring_segments, 3)
	"""
	polyline = np.asarray(polyline)
	n = len(polyline)

	rings = []

	# Precompute tangents
	if np.allclose(polyline[0], polyline[-1], atol=1e-6):	#snap end points
		tangents = geo.polyline.tangents(polyline[:-1], closed=True)
		tangents.append(tangents[0])
		#print(tangents)
	else:
		tangents = geo.polyline.tangents(polyline, closed=False)

	# Build rings
	prev_normal = None

	for i in range(n):
		t = tangents[i]

		# Find a stable normal vector perpendicular to tangent
		if prev_normal is None:
			# pick any vector not parallel to t
			tmp = np.array([1, 0, 0])
			if abs(np.dot(tmp, t)) > 0.9:
				tmp = np.array([0, 1, 0])
			normal = npx.normalize(np.cross(t, tmp))
		else:
			# make normal smooth by projecting previous one onto plane ⟂ t
			normal = prev_normal - t * np.dot(prev_normal, t)
			normal = npx.normalize(normal)

			# If degenerate, choose a new one
			if np.linalg.norm(normal) < 1e-6:
				tmp = np.array([1, 0, 0])
				if abs(np.dot(tmp, t)) > 0.9:
					tmp = np.array([0, 1, 0])
				normal = npx.normalize(np.cross(t, tmp))

		# Binormal
		binormal = npx.normalize(np.cross(t, normal))

		# Create the ring (circle)
		ring = []
		for k in range(ring_segments):
			angle = 2.0 * np.pi * k / ring_segments
			offset = (normal * np.cos(angle) + binormal * np.sin(angle)) * ring_radius
			ring.append(polyline[i] + offset)
		ring = np.array(ring)

		rings.append(ring)
		prev_normal = normal

	return rings

def tube(polyline, ring_radius=1.0, ring_segments=16):
	rings = generate_rings(polyline, ring_radius=ring_radius, ring_segments=ring_segments)
	result = geo.Mesh()
	for x in rings:
		result.vertices.extend(x)
	primitives = [len(rings[0])] * len(rings)
	result.faces = geo.enlongated_faces(*primitives)
	return result

def torus(n, r=1.0, ring_radius=.25):
	polyline = [np.array([x[0], 0.0, x[1]]) for x in npx.on_arc(n, start=0.0, size=math.pi * 2)]
	#print(polyline)
	return tube(polyline, ring_radius=ring_radius, ring_segments=n)

def grid_triangulation(cell_count, closed=[False, False]):
	cell_count = cell_count.astype(int)
	def dt(index):
		return fan_triangulate([
			npx.column_major_order(cell_count, np.array([index[0], index[1]])),
			npx.column_major_order(cell_count, np.array([index[0], index[1] + 1])),
			npx.column_major_order(cell_count, np.array([index[0] + 1, index[1] + 1])),
			npx.column_major_order(cell_count, np.array([index[0] + 1, index[1]]))
			])
	shape = [cell_count[i] - (0 if closed[i] else 1) for i in range(2)]
	return flatten(np.fromfunction(lambda *index: dt(index), shape), 2)

def plane(offset=np.zeros(2), cell_size=np.ones(2), cell_count=np.ones(2) * 10):
	#vertices = flatten(Map([0, 0, 0], stop=[cell_count[0], 1, cell_count[1]], step=None, func=lambda *index: np.array(index)), 2)
	vertices = [np.array([x.min[0], 0, x.min[1]]) for x in npx.grid(offset=offset, cell_size=cell_size).cells(stop=cell_count)]
	uvs = [x[[0, 2]] / (cell_size * cell_count) for x in vertices]
	#print(uvs)
	triangles = grid_triangulation(cell_count, closed=[False, False])
	#print(triangles)
	return Mesh(vertices, triangles, uvs)

def cube(position, size, pivot=np.ones(3) * 0.5):

	o = position - size * pivot

	vertices = []

	for i in range(8):
		x = (i >> 0) & 1
		y = (i >> 1) & 1
		z = (i >> 2) & 1

		vertices.append(o + size * np.array([x, y, z]))

	faces = [
		[0, 1, 3, 2],  # z = 0
		[4, 5, 7, 6],  # z = 1
		[0, 1, 5, 4],  # y = 0
		[2, 3, 7, 6],  # y = 1
		[0, 2, 6, 4],  # x = 0
		[1, 3, 7, 5],  # x = 1
	]

	return Mesh(vertices, faces)



def grid_uv(cell_count, swizzle=[1, 0]):
	return npx.cartesian_product([npx.subdivide(0., 1., cell_count[i]) for i in range(2)], swizzle=swizzle)

def polygon_uv(vertices):
	bbox = npx.aabb(vertices)
	return np.array([bbox.normalize_point(x) for x in vertices])




