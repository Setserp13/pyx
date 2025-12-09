import math
import numpy as np
import pyx.numpyx as npx
import pyx.numpyx_geo as geo
from pyx.numpyx_geo import Mesh
from pyx.mat.transform import Node3D
import struct
import json
from pyx.collectionsx import flatten
import copy
import pyx.mat.mat as mat
import pyx.rex as rex

def fan_triangulate(indices):
	return [[indices[0], indices[i], indices[i+1]] for i in range(1, len(indices) - 1)]

class glb:
	ARRAY_BUFFER = 34962	#dados de vértices (positions, normals, tangents, UVs, colors, joints, weights etc.)
	ELEMENT_ARRAY_BUFFER = 34963	#índices (faces, triângulos)	

	def __init__(self):
		self.buffer_views = []

	def add_buffer_view(self, bytes, target, componentType, type):
		self.buffer_views.append({"bytes": bytes, "target": target, "componentType": componentType, "type": type})

	@property
	def buffer(self):
		result = bytearray()
		for x in self.buffer_views:
			result += x["bytes"]
		return result

	@property
	def bufferViews(self):
		result = []
		byteOffset = 0
		for x in self.buffer_views:
			byteLength = len(x["bytes"])
			view = {
					"buffer": 0,	#Indica qual buffer esse trecho pertence. No GLB, normalmente sempre é 0
					"byteOffset": byteOffset,
					"byteLength": byteLength
				}
			if not x["target"] is None:
				view["target"] = x["target"]
			result.append(view)
			byteOffset += byteLength
		return result

	@property
	def accessors(self):
		result = []
		for i, x in enumerate(self.buffer_views):
			result.append(
				{
					"bufferView": i,
					"byteOffset": 0,
					"componentType": x["componentType"],
					"count": len(x["bytes"]) // ({"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[x["type"]] * 4),
					"type": x["type"]
				}
			)
		return result

	def pack(gltf, bin, filename):
		gltf = rex.lpad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), 4, b" ")	# Convert JSON to binary (must be padded to 4-byte)
		bin = rex.lpad(bin, 4, b"\x00")	# Pad BIN chunk to 4 bytes

		# GLB header
		magic = 0x46546C67			# 'glTF'
		version = 2

		total_length = 12 + (8 + len(gltf)) + (8 + len(bin))

		with open(filename, "wb") as f:
			f.write(struct.pack("<III", magic, version, total_length))	# Header
			f.write(struct.pack("<I4s", len(gltf), b"JSON"))	# JSON chunk
			f.write(gltf)
			f.write(struct.pack("<I4s", len(bin), b"BIN\0"))	# BIN chunk
			f.write(bin)
		print("Saved", filename)


class Skin():
	def __init__(self, joints=None):
		self.joints = [] if joints is None else joints	# list of Node3D-like transforms




class SkinnedMesh(Mesh, Node3D):
	def __init__(self, vertices, faces, uvs=None, skin=None, joints=None, weights=None, **kwargs):
		Mesh.__init__(self, vertices, faces, uvs)
		Node3D.__init__(self, **kwargs)
		self.skin = skin
		self.joints = [] if joints is None else joints		# list[ list[int] ]
		self.weights = [] if weights is None else weights	# list[ list[float] ]

	#each mesh has a skeleton attached to it
	#in glb, you references joints in mesh by their relative indices to the skin they belong
	def to_glb(roots, filename):	#self, filename):
		if not filename.lower().endswith(".glb"):
			filename += ".glb"

		
		new_roots = []
		for x in roots:
			parent = Node3D()
			new_roots.append(parent)
			parent.append(x)
			for y in x.skin.joints:
				if y.parent is None:
					parent.append(y)
		roots = new_roots

		#nodes3d = flatten([[x, *x.skin.joints] for x in roots])
		nodes3d = flatten([[x, *x.descendants()] for x in roots])
		for i, x in enumerate(nodes3d):
			x.index = i
			x.name = f'Node ({i})'
		
		# BUILD BINARY BUFFER FOR GEOMETRY
		obj = glb()

		meshes = []
		skins = []
		nodes = []
		for x in nodes3d:
			node = {'name': x.name, 'children': [y.index for y in x.children], 'translation': x.position.tolist()}
			nodes.append(node)
			if not isinstance(x, SkinnedMesh):
				continue
			mesh = x
			primitive = {
				"attributes": {},
				"indices": 1,
				"mode": 4
				}
			node['mesh'] = len(meshes)
			primitive['attributes']['POSITION'] = len(obj.buffer_views)	#accessor, but this case it works 'cause len(bufferViews) == len(accessors)
			obj.add_buffer_view(np.asarray(mesh.vertices, dtype="<f4").tobytes(), 34962, 5126, "VEC3")
			primitive['indices'] = len(obj.buffer_views)
			obj.add_buffer_view(np.asarray(mesh.faces, dtype="<u4").tobytes(), 34963, 5125, "SCALAR")	#faces must be triangles
			if len(mesh.uvs) > 0:
				primitive['attributes']['TEXCOORD_0'] = len(obj.buffer_views)
				obj.add_buffer_view(np.asarray(mesh.uvs, dtype="<f4").tobytes(), 34962, 5126, "VEC2")
			if hasattr(mesh, 'skin') and len(mesh.skin.joints) > 0:
				node['skin'] = len(skins)
				skins.append({"joints": [y.index for y in mesh.skin.joints], "inverseBindMatrices": len(obj.buffer_views)})
				#print([(x.inverse_TRS().T) for x in mesh.skin.joints])
				obj.add_buffer_view(np.asarray([x.inverse_TRS().T for x in mesh.skin.joints], dtype="<f4").tobytes(), None, 5126, "MAT4")	#34962, 5126, "MAT4")
				if len(mesh.joints) > 0:
					primitive['attributes']['JOINTS_0'] = len(obj.buffer_views)
					obj.add_buffer_view(np.asarray(mesh.joints, dtype="<u4").tobytes(), 34962, 5125, "VEC4")
				if len(mesh.weights) > 0:
					primitive['attributes']['WEIGHTS_0'] = len(obj.buffer_views)
					obj.add_buffer_view(np.asarray(mesh.weights, dtype="<f4").tobytes(), 34962, 5126, "VEC4")
			meshes.append({
				"primitives": [
					primitive
					]
				})

		#print(meshes)		

		gltf = {
			"asset": { "version": "2.0" },
			"buffers": [
				{ "byteLength": len(obj.buffer) }
			],
			"bufferViews": obj.bufferViews,
			"accessors": obj.accessors,
			"meshes": meshes,
			"skins": skins,
			"nodes": nodes,
			"scenes": [ { "nodes": [0] } ],
			"scene": 0
		}

		glb.pack(gltf, obj.buffer, filename)

#---
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
