import numpy as np
import pyx.numpyx as npx
from pyx.generic.node import Node
import functools
import itertools
import math
from itertools import combinations
from functools import reduce

"""
Use R.T (transposta) — é mais rápido e numericamente mais estável que np.linalg.inv.
Se a matriz não for ortogonal (por exemplo contém escala, cisalhamento ou erro numérico), R.T não será a inversa — aí use np.linalg.inv.
Para rotações representadas por quaternions, a inversa (rotação oposta) é o conjugado do quaternion normalizado; ao converter para matriz, a transposta continua sendo a inversa.
"""

def homogenize(M):
	"""Embed a square matrix in homogeneous coordinates."""
	M = np.asarray(M)

	if M.ndim != 2 or M.shape[0] != M.shape[1]:
		raise ValueError("M must be a square matrix")

	H = np.eye(M.shape[0] + 1, dtype=M.dtype)
	H[:-1, :-1] = M

	return H


def R2(angle):
	c, s = np.cos(angle), np.sin(angle)
	return np.array([[c, s], [-s, c]])

def basic_rotation_matrix(dim, plane, angle):	#Return a basic n-dimensional rotation matrix by an angle θ in the [i, j] plane
	M = np.eye(dim)
	i, j = plane
	M[np.ix_([i, j], [i, j])] = R2(angle)
	return M

def R(angles):
	# Solve n(n-1)/2 = len(angles)
	k = len(angles)
	n = int((1 + math.sqrt(1 + 8*k)) / 2)

	axes = range(n)
	planes = combinations(axes, 2)

	rotations = [
		basic_rotation_matrix(n, plane, angle)
		for angle, plane in zip(angles, planes)
	]

	return reduce(lambda A, B: A @ B, rotations)

def S(scale):	#scale matrix
	scale = np.asarray(scale, dtype=float)
	return np.diag(scale)

def T(offset):	#translation matrix
	offset = np.asarray(offset, dtype=float)
	n = len(offset)

	M = np.eye(n + 1)
	M[:n, n] = offset

	return M

def skew(v):
	x, y, z = v
	return np.array([
		[0, -z, y],
		[z, 0, -x],
		[-y, x, 0]
	])

"""def R3(q):
	x, y, z, w = np.asarray(q) / np.linalg.norm(q)
	v = np.array([x, y, z])

	return np.eye(3) + 2 * w * skew(v) + 2 * skew(v) @ skew(v)"""

class Matrix:

	@staticmethod
	def decompose(matrix):	#decompose T R S
		dim = matrix.shape[0] - 1
		position = matrix[:dim, dim].copy()
	
		basis = matrix[:dim, :dim]
		u, singular_values, vt = np.linalg.svd(basis)
	
		rotation = u @ vt
		scale = np.diag(rotation.T @ basis).copy()
	
		# Keep the rotation matrix orientation-preserving.
		if np.linalg.det(rotation) < 0:
			u[:, -1] *= -1
			rotation = u @ vt
			scale = np.diag(rotation.T @ basis).copy()
	
		return position, rotation, scale	#(dim,), (dim, dim), (dim,)


class quaternion(np.ndarray):
	def __new__(cls, input_array):
		obj = np.asarray(input_array).view(cls)
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return

	@property
	def conjugate(q):
		x, y, z, w = q
		return quaternion([-x, -y, -z, w])

	@property
	def inverse(q):	#For a unit quaternion, your conjugate is also its inverse.
		return q.conjugate / np.dot(q, q)
	
	def rotate(self, v):
		# v é um vetor 3D (lista/tupla/np.array)
		q = np.array(self[:3])
		t = 2 * np.cross(q, v)
		return np.array(v) + self[3] * t + np.cross(q, t)

	def to_euler(q):
		x, y, z, w = q
	
		return np.array([
			np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)),
			np.arcsin(np.clip(2*(w*y - z*x), -1, 1)),
			np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
		])

	@staticmethod
	def from_euler(euler):
		ex, ey, ez = euler
	
		s = np.sin(np.array(euler) / 2)
		c = np.cos(np.array(euler) / 2)
		sx, sy, sz = s
		cx, cy, cz = c
	
		return quaternion([
			sx*cy*cz - cx*sy*sz,
			cx*sy*cz + sx*cy*sz,
			cx*cy*sz - sx*sy*cz,
			cx*cy*cz + sx*sy*sz
		])
	
	@staticmethod
	def multiply(q1, q2):
		v1, w1 = q1[:3], q1[3]
		v2, w2 = q2[:3], q2[3]
	
		return np.r_[
			w1*v2 + w2*v1 + np.cross(v1, v2),
			w1*w2 - np.dot(v1, v2)
		]
	
	@staticmethod
	def from_matrix(M):
		# Accept 4×4 or 3×3
		if M.shape == (4, 4):
			m = M[:3, :3]
		else:
			m = M
	
		trace = m[0,0] + m[1,1] + m[2,2]
	
		if trace > 0:
			s = 0.5 / np.sqrt(trace + 1.0)
			w = 0.25 / s
			x = (m[2,1] - m[1,2]) * s
			y = (m[0,2] - m[2,0]) * s
			z = (m[1,0] - m[0,1]) * s
	
		elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
			s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
			w = (m[2,1] - m[1,2]) / s
			x = 0.25 * s
			y = (m[0,1] + m[1,0]) / s
			z = (m[0,2] + m[2,0]) / s
	
		elif m[1,1] > m[2,2]:
			s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
			w = (m[0,2] - m[2,0]) / s
			x = (m[0,1] + m[1,0]) / s
			y = 0.25 * s
			z = (m[1,2] + m[2,1]) / s
	
		else:
			s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
			w = (m[1,0] - m[0,1]) / s
			x = (m[0,2] + m[2,0]) / s
			y = (m[1,2] + m[2,1]) / s
			z = 0.25 * s
	
		return quaternion([x, y, z, w])

	def to_matrix(q):
		x, y, z, w = np.asarray(q) / np.linalg.norm(q)
		v = np.array([x, y, z])
	
		return np.eye(3) + 2 * w * skew(v) + 2 * skew(v) @ skew(v)
	
	@staticmethod
	def from_axis_angle(axis, angle):
		axis = npx.normalize(axis)
		s = np.sin(angle * 0.5)
		return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)], dtype=float)

	@staticmethod
	def align_vector(local_vec, world_vec):
		# Normalizar
		a = npx.normalize(local_vec)
		b = npx.normalize(world_vec)
	
		# Caso especial: já alinhados
		dot = np.dot(a, b)
		if dot > 0.999999:
			return np.array([1, 0, 0, 0], float)  # identidade
	
		# Caso especial: opostos → rotação 180° em qualquer eixo perpendicular
		if dot < -0.999999:
			axis = npx.normalize(np.cross(a, np.array([1, 0, 0])))
			if np.linalg.norm(axis) < 1e-6:
				axis = npx.normalize(np.cross(a, np.array([0, 1, 0])))
			return quaternion.from_axis_angle(axis, np.pi)
	
		axis = npx.normalize(np.cross(a, b))	# Eixo = perpendicular entre A e B
		angle = np.arccos(np.clip(dot, -1.0, 1.0))	# Ângulo = distância angular entre os vetores
		return quaternion.from_axis_angle(axis, angle)	# Quaternion final



class Basis(np.ndarray):
	def __new__(cls, matrix):
		matrix = np.asarray(matrix, dtype=float)

		if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
			raise ValueError("Basis must be a square matrix")

		return matrix.view(cls)

	def __array_finalize__(self, obj):
		pass

	@property
	def ndim(self): return self.shape[0]

	@classmethod
	def identity(cls, n): return cls(np.eye(n))

	@property
	def matrix(self): return np.asarray(self)

	@property
	def rotation(self):
		return self._decompose()[0]

	@property
	def _upper(self):
		return self._decompose()[1]

	@property
	def scale(self):
		return np.diag(self._upper)

	@property
	def shear(self):
		U = self._upper
		return np.array([
			U[i, j] / U[j, j]
			for i in range(self.ndim)
			for j in range(i + 1, self.ndim)
		])

	@scale.setter
	def scale(self, value):
		self[:] = self.compose(scale=value, shear=self.shear, rotation=self.rotation)
	
	@shear.setter
	def shear(self, value):
		self[:] = self.compose(scale=self.scale, shear=value, rotation=self.rotation)

	@rotation.setter
	def rotation(self, value):
		self[:] = self.compose(scale=self.scale, shear=self.shear, rotation=value)
	
	def _decompose(self):
		Q, R = np.linalg.qr(self.matrix)	#QR decomposition
	
		signs = np.where(np.diag(R) < 0, -1., 1.)
	
		Q = Q @ np.diag(signs)
		R = np.diag(signs) @ R
	
		# Ensure Q is a proper rotation. Fix improper rotation (reflection)
		if np.linalg.det(Q) < 0:
			Q[:, -1] *= -1
			R[-1, :] *= -1
	
		# Check for singular or nearly singular bases.
		tol = np.finfo(R.dtype).eps * np.linalg.norm(R, ord=np.inf)
	
		if np.any(np.abs(np.diag(R)) <= tol):
			raise ValueError("Cannot decompose a singular (non-invertible) basis")
	
		return Q, R	#rotation, (scale @ shear @ reflection)

	@classmethod
	def compose(cls, scale, shear=None, rotation=None):
		scale = np.asarray(scale, dtype=float)
	
		if scale.ndim != 1:
			raise ValueError("Scale must be a 1D array")
	
		n = len(scale)
		expected = n * (n - 1) // 2
	
		if shear is None:
			shear = np.zeros(expected)
		else:
			shear = np.asarray(shear, dtype=float)
	
		if shear.shape != (expected,):
			raise ValueError(f"Shear must contain {expected} values")
	
		if rotation is None:
			rotation = np.eye(n)
		else:
			rotation = np.asarray(rotation, dtype=float)
	
		if rotation.shape != (n, n):
			raise ValueError("Rotation must be an n x n matrix")
	
		H = np.eye(n)
		k = 0
	
		for i in range(n):
			for j in range(i + 1, n):
				H[i, j] = shear[k]
				k += 1
	
		return cls(rotation @ H @ np.diag(scale))

	def __repr__(self):
		return (
			f"Basis(\n"
			f"\tmatrix=\n{self.matrix},\n"
			f"\tscale={self.scale},\n"
			f"\tshear={self.shear},\n"
			f"\trotation=\n{self.rotation}\n"
			f")"
		)







class Transform(Node):
	def __init__(self, position, rotation, scale, **kwargs):
		super().__init__(**kwargs)
		self.position = np.array(position)
		self._rotation = rotation
		self.scale = np.array(scale)

	@property
	def ndim(self): return len(self.position)

	@classmethod
	def identity(cls, ndim):
		return cls(
			position=np.zeros(ndim),
			rotation=np.eye(ndim),
			scale=np.ones(ndim)
		)
	
	@property
	def T(self): return T(self.position)

	#@property
	#def R(self): pass
	@property
	def R(self): return homogenize(self._rotation)
	
	@property
	def S(self): return homogenize(S(self.scale))

	@property
	def global_TRS(self):
		return self.TRS if self.parent is None else self.parent.global_TRS @ self.TRS
		#return functools.reduce(lambda acc, x: acc @ x, [x.TRS for x in reversed([self] + self.ancestors())])

	@property
	def inv_global_TRS(self): return np.linalg.inv(self.global_TRS)

	def TRS_to(self, parent):
		result = self.TRS
		node = self.parent
	
		while node is not None and node != parent:
			result = node.TRS @ result
			node = node.parent
	
		if node != parent:
			raise ValueError("The specified node is not an ancestor")
	
		return result

	
	def inv_TRS_to(self, parent): return np.linalg.inv(self.TRS_to(parent))

	@property
	def TRS(self):	#local transformation matrix
		#print(self.T, self.R, self.S)
		return self.T @ self.R @ self.S

	@property
	def inv_TRS(self): #local inverse transformation matrix
		return np.linalg.inv(self.TRS)
		#return np.linalg.inv(self.S) @ np.linalg.inv(self.R) @ np.linalg.inv(self.T)

	def to_local(self, point):
		p = np.append(point, 1)
		return (self.inv_global_TRS @ p)[:self.ndim]

	def to_global(self, point):
		p = np.append(point, 1)
		return (self.global_TRS @ p)[:self.ndim]

	@property
	def global_position(self): return self.global_TRS[:self.ndim, -1]

	@global_position.setter
	def global_position(self, value):
		if self.parent is None:
			self.position = np.array(value)
		else:
			self.position = self.parent.to_local(value)
	
	@property
	def global_scale(self):
		return np.linalg.norm(self.basis, axis=0)
	
	@global_scale.setter
	def global_scale(self, value):
		value = np.array(value)
		if self.parent is None:
			self.scale = value
		else:
			self.scale = value / self.parent.global_scale

	"""@property
	def basis(self):	# BASIS (n×n matrix of world axes) -> upper-left n×n
		return self.global_TRS[:self.ndim, :self.ndim]

	@property
	def local_basis(self):
		return self.TRS[:self.ndim, :self.ndim]"""

	@property
	def basis(self):
		return self.TRS[:self.ndim, :self.ndim]

	@property
	def global_basis(self):	# BASIS (n×n matrix of world axes) -> upper-left n×n
		return self.global_TRS[:self.ndim, :self.ndim]
	
	@property
	def matrix(self): return self.TRS
	
	@property
	def global_matrix(self): return self.global_TRS

	@matrix.setter
	def matrix(self, value):
		value = np.asarray(value, dtype=float)
	
		n = self.ndim
	
		if value.shape != (n + 1, n + 1):
			raise ValueError("Matrix must be an (n+1) x (n+1) matrix")
	
		self.position = value[:n, n]
	
		basis = Basis(value[:n, :n])
	
		self.scale = basis.scale
		self.rotation = basis.rotation

	@global_matrix.setter
	def global_matrix(self, value):
		parent = self.attrib.get("parent")
	
		if parent is not None:
			value = np.linalg.inv(parent.global_TRS) @ value
	
		self.matrix = value

	
	@property
	def rotation_matrix(self):
		raise NotImplementedError
	
	@rotation_matrix.setter
	def rotation_matrix(self, value):
		raise NotImplementedError
	
	def _reparent(self, objects, operation):	#WORLD POSITION STAYS
		objects = list(objects)
		global_trs = [x.global_TRS.copy() for x in objects]
	
		operation()
	
		for x, trs in zip(objects, global_trs):
			parent = x.parent
			local_trs = np.linalg.inv(parent.global_TRS) @ trs if parent else trs
			x.position, x.rotation_matrix, x.scale = Matrix.decompose(local_trs)
	
	
	def set_parent_global(self, parent):
		self._reparent([self], lambda: Node.set_parent(self, parent))
	
	
	def append_global(self, child):
		self._reparent([child], lambda: Node.append(self, child))
	
	
	def extend_global(self, children):
		children = list(children)
		self._reparent(children, lambda: Node.extend(self, children))

	def _fit_matrix(self, value):
		if isinstance(value, Transform):
			value = value.matrix
	
		value = np.asarray(value, dtype=float)
		n = self.ndim + 1
	
		if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] > n:
			raise ValueError("Incompatible matrix dimensions")
	
		if value.shape != (n, n):
			matrix = np.eye(n)
			matrix[:value.shape[0], :value.shape[1]] = value
			return matrix
	
		return value
	
	
	def __matmul__(self, other):
		result = type(self)()
		result.matrix = self.matrix @ self._fit_matrix(other)
		return result
	
	
	def __imatmul__(self, other):
		self.matrix = (self @ other).matrix
		return self
	
	
	def __rmatmul__(self, other):
		result = type(self)()
		result.matrix = self._fit_matrix(other) @ self.matrix
		return result

class Node2D(Transform):	#Node):
	def __init__(self, position=np.zeros(2), rotation=0.0, scale=np.ones(2), **kwargs):
		#super().__init__(position, rotation, scale, **kwargs)
		super().__init__(position, R2(rotation), scale, **kwargs)

	@property
	def rotation(self):
		return np.arctan2(self._rotation[1, 0], self._rotation[0, 0])
	@rotation.setter
	def rotation(self, value):
		self._rotation = R2(value)
	#@property
	#def R(self): return homogenize(R2(self.rotation))

	@classmethod
	def from_matrix(cls, M):
		t = decompose_trs_shear(M)
		rotation = math.atan2(t['R'][1, 0], t['R'][0, 0])
		#print(rotation)
		return cls(position=t['T'], rotation=rotation, scale=t['S'])

	@property
	def rotation_matrix(self): return self.R
	
	@rotation_matrix.setter
	def rotation_matrix(self, value):
		self.rotation = np.arctan2(value[1, 0], value[0, 0])




class Node3D(Transform):	#Node):):
	def __init__(self, position=np.zeros(3), rotation=quaternion([0, 0, 0, 1]), scale=np.ones(3), **kwargs):
		super().__init__(position, rotation, scale, **kwargs)

	"""@property
	def euler(self): return self.rotation.to_euler()
	@euler.setter
	def euler(self, value): self.rotation = quaternion.from_euler(value)

	@property
	def R(self): return homogenize(R3(self.rotation))"""

	@property
	def rotation(self): return quaternion.from_matrix(self._rotation)
	@rotation.setter
	def rotation(self, value): self._rotation = value.to_matrix()
	
	@property
	def euler(self): return self.quaternion.to_euler()
	@euler.setter
	def euler(self, value): self.rotation = quaternion.from_euler(value)





def decompose_trs_shear(M, eps=1e-8):
	"""
	Decompose a homogeneous affine matrix into:
	- translation vector T
	- rotation matrix R
	- scale vector S
	- shear matrix Sh

	Works in any dimension.
	"""
	M = np.asarray(M, dtype=float)
	n = M.shape[0] - 1

	if M.shape != (n + 1, n + 1):
		raise ValueError("Matrix must be homogeneous (N+1 x N+1)")

    
	T = M[:-1, -1].copy()	# 1. Translation

	A = M[:-1, :-1]	# 2. Linear part

	ATA = A.T @ A	# 3. Polar decomposition

	# Eigen-decomposition of symmetric matrix
	eigvals, eigvecs = np.linalg.eigh(ATA)
	eigvals = np.maximum(eigvals, eps)

	H = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
	R = A @ np.linalg.inv(H)

	# Fix improper rotation (reflection)
	if np.linalg.det(R) < 0:
		R[:, 0] *= -1
		H[0, :] *= -1

	S = np.diag(H).copy()	# 4. Scale (diagonal of H)

	# 5. Shear matrix
	Sh = H.copy()
	for i in range(n):
		if abs(S[i]) > eps:
			Sh[i, :] /= S[i]

	np.fill_diagonal(Sh, 1.0)

	return {
		"T": T,
		"R": R,
		"S": S,
		"ShearMatrix": Sh
	}

def compose_trs_shear(T, R, S, Sh):
	n = len(T)

	Sm = np.diag(S)	# Scale matrix

	H = Sm @ Sh	# Shear+scale

	A = R @ H	# Linear part

	# Homogeneous matrix
	M = np.eye(n + 1)
	M[:-1, :-1] = A
	M[:-1, -1] = T

	return M

def random_rotation(n):
	A = np.random.randn(n, n)
	Q, _ = np.linalg.qr(A)
	if np.linalg.det(Q) < 0:
		Q[:, 0] *= -1
	return Q


def test_decompose_trs_shear():
	np.random.seed(42)

	for n in [2, 3, 5, 8]:
		for _ in range(50):
			# Random components
			T = np.random.uniform(-10, 10, size=n)
			R = random_rotation(n)
			S = np.random.uniform(0.5, 3.0, size=n)

			# Random shear (unit diagonal)
			Sh = np.eye(n)
			Sh += np.random.uniform(-0.3, 0.3, size=(n, n))
			np.fill_diagonal(Sh, 1.0)

			M = compose_trs_shear(T, R, S, Sh)	# Compose

			out = decompose_trs_shear(M)	# Decompose

			# Recompose
			M2 = compose_trs_shear(out["T"], out["R"], out["S"], out["ShearMatrix"])

			assert np.allclose(M, M2, atol=1e-6), f"Failed in {n}D"	# Assertions

	print("All decomposition tests passed ✅")

#test_decompose_trs_shear()

