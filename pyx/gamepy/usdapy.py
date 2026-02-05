import numpy as np
import math
import pyx.osx as osx
from pyx.numpyx_geo import Mesh
from pyx.mat.transform import Node3D
from pyx.collectionsx import flatten
from pyx.gamepy.material import Material
import pyx.rex as rex

def astuple(x):
	"""
	Convert scalars / vectors / arrays into USD-friendly tuples.
	"""
	if isinstance(x, np.ndarray):
		if x.ndim == 1:
			return tuple(float(v) for v in x)
		return [tuple(float(v) for v in row) for row in x]

	if isinstance(x, (list, tuple)):
		if len(x) == 0:
			return ()
		if isinstance(x[0], (list, tuple, np.ndarray)):
			return [tuple(float(v) for v in row) for row in x]
		return tuple(float(v) for v in x)

	if isinstance(x, (int, float)):
		return float(x)

	raise TypeError(f"Unsupported type: {type(x)}")

def block(lines, indent=0):
	result = '\t' * indent + '{\n'
	result += '\n'.join(['\t' * (indent + 1) + x for x in lines])
	result += f'\n' + '\t' * indent + '}'
	return result

def Mesh_to_usda(self, indent=0):
	result = '\t' * indent + f'def Mesh "{self.name}"'	#header
	lines = [
		f'point3f[] points = {astuple(self.vertices)}',
		f'int[] faceVertexCounts = {[len(x) for x in self.faces]}',
		f'int[] faceVertexIndices = {flatten(self.faces, 1)}'
	]
	if hasattr(self, 'tracks'):
		for x in self.tracks:
			lines.append(x.to_usda(indent))
	if self.normals_interpolation:
		match self.normals_interpolation:
			case 'constant':
				normals = self.normal
			case 'vertex':
				normals = self.vertex_normals
			case 'uniform':
				normals = self.face_normals
			case 'face_varying':
				normals = self.flat_corner_normals
		#print(type(normals[0]))
		lines.append(f'normal3f[] normals = {astuple(normals)}')
		lines.append(f'uniform token normalsInterpolation = "{rex.snake_to_camel(self.normals_interpolation)}"')
	if self.uvs:	#not None
		lines.append(f'float2[] primvars:st = {astuple(self.uvs)}')
		lines.append(f'uniform token primvars:st:interpolation = "{rex.snake_to_camel(self.uvs_interpolation)}"')
	if self.colors:	#not None
		lines.append(f'color3f[] primvars:displayColor = {astuple(self.colors)}')
		lines.append(f'uniform token primvars:displayColor:interpolation = "{rex.snake_to_camel(self.colors_interpolation)}"')
	lines.append(f'bool doubleSided = {"true" if self.double_sided else "false"}')
	"""lines.append(f'uniform token subdivisionScheme = "{self.subdivision_scheme}"')
	lines.append(f'token visibility = "{self.visibility}"')
	lines.append(f'uniform token purpose = "{self.purpose}"')"""
	result += '\n' + block(lines, indent)
	return result

def Node3D_to_usda(self, indent=0):
	result = '\t' * indent + f'def Xform "{self.name}"'	#header
	lines = [
		f'double3 xformOp:translate = {astuple(self.position)}',
		f'double3 xformOp:scale = {astuple(self.scale)}',
		f'double3 xformOp:rotateXYZ = {astuple(self.euler)}',
		'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]'
	]
	"""lines.append(f'token visibility = "{self.visibility}"')
	lines.append(f'uniform token purpose = "{self.purpose}"')
	lines.append(f'bool resetXformStack = {self.reset_xform_stack}')"""

	for x in self.children:
		#lines.append(x.to_usda(indent + 1))
		lines.extend(x.to_usda(indent).split('\n'))

	result += '\n' + block(lines, indent)
	return result

def Material_to_usda(self):
	result = '\t' * indent + f'def Material "{self.name}"'	#header
	lines = [
		f'token outputs:surface.connect = </{self.name}/PreviewSurface.outputs:surface>',
		f'def Shader "PreviewSurface"',
		'{',
		'\tuniform token info:id = "UsdPreviewSurface"',
		f'\tcolor3f inputs:diffuseColor = ({self.albedo[0]}, {self.albedo[1]}, {self.albedo[2]})',
		f'\tfloat inputs:metallic = {self.metallic}',
		f'\tfloat inputs:roughness = {self.roughness}',
		'\ttoken outputs:surface',
		'}',
	]
	result += '\n' + block(lines, indent)
	return result

Mesh.to_usda = Mesh_to_usda
Node3D.to_usda = Node3D_to_usda
Material.to_usda = Material_to_usda
