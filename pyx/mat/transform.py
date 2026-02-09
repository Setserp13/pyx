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

def to_homogeneous(matrix):
	dim = len(matrix)
	result = np.eye(dim+1)
	result[:dim, :dim] = matrix
	return result



def R2(theta):
	c = math.cos(theta)
	s = math.sin(theta)
	return np.array([[c, -s],
					 [s,  c]])

def basic_rotation_matrix(theta, plane, dim):
	M = np.eye(dim)
	i, j = plane
	M[np.ix_([i, j], [i, j])] = R2(theta)
	return M

def R(angles):
	# Solve n(n-1)/2 = len(angles)
	k = len(angles)
	n = int((1 + math.sqrt(1 + 8*k)) / 2)

	axes = range(n)
	planes = combinations(axes, 2)

	rotations = [
		basic_rotation_matrix(angle, plane, n)
		for angle, plane in zip(angles, planes)
	]

	return reduce(lambda A, B: A @ B, rotations)






class Matrix:
	def S(vector, homogeneous=True): #scaling
		if homogeneous:
			vector = np.append(vector, 1)
		return np.array([(npx.ei(i, len(vector)) * x).tolist() for i, x in enumerate(vector)])

	def T(vector): #translation
		n = len(vector)
		result = np.eye(n + 1)
		result[:n, -1] = vector
		return result

	def R2(theta): #rotation 2D
		return np.array([
			[np.cos(theta), -np.sin(theta), 0],
			[np.sin(theta),  np.cos(theta), 0],
			[0, 0, 1]
		])

	@staticmethod
	def R3(q):
		q = q / np.linalg.norm(q)
		x, y, z, w = q

		xx, yy, zz = x*x, y*y, z*z
		xy, xz, yz = x*y, x*z, y*z
		wx, wy, wz = w*x, w*y, w*z

		R = np.array([
			[1 - 2*(yy + zz),   2*(xy - wz),     2*(xz + wy)],
			[2*(xy + wz),       1 - 2*(xx + zz), 2*(yz - wx)],
			[2*(xz - wy),       2*(yz + wx),     1 - 2*(xx + yy)]
		])

		# ⬇ Convert to 4×4 homogeneous matrix
		"""R4 = np.eye(4)
		R4[:3, :3] = R
		return R4"""
		return to_homogeneous(R)

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

	def rotate(self, v):
		# v é um vetor 3D (lista/tupla/np.array)
		vx, vy, vz = v

		qx, qy, qz, qw = self

		# t = 2 * cross(q.xyz, v)
		tx = 2 * (qy * vz - qz * vy)
		ty = 2 * (qz * vx - qx * vz)
		tz = 2 * (qx * vy - qy * vx)

		# v' = v + qw * t + cross(q.xyz, t)
		vpx = vx + qw * tx + (qy * tz - qz * ty)
		vpy = vy + qw * ty + (qz * tx - qx * tz)
		vpz = vz + qw * tz + (qx * ty - qy * tx)

		return np.array([vpx, vpy, vpz], float)

	def to_euler(q):
		x, y, z, w = q
	
		# Rotation around X
		sin_x = 2 * (w*x + y*z)
		cos_x = 1 - 2 * (x*x + y*y)
		ex = np.arctan2(sin_x, cos_x)
	
		# Rotation around Y
		sin_y = 2 * (w*y - z*x)
		sin_y = np.clip(sin_y, -1, 1)
		ey = np.arcsin(sin_y)
	
		# Rotation around Z
		sin_z = 2 * (w*z + x*y)
		cos_z = 1 - 2 * (y*y + z*z)
		ez = np.arctan2(sin_z, cos_z)
	
		return np.array([ex, ey, ez])

	@staticmethod
	def from_euler(euler):
		ex, ey, ez = euler
	
		# Half angles
		hx = ex * 0.5
		hy = ey * 0.5
		hz = ez * 0.5
	
		sx = np.sin(hx)
		cx = np.cos(hx)
		sy = np.sin(hy)
		cy = np.cos(hy)
		sz = np.sin(hz)
		cz = np.cos(hz)
	
		# Combine into quaternion (x, y, z, w)
		x = sx*cy*cz - cx*sy*sz
		y = cx*sy*cz + sx*cy*sz
		z = cx*cy*sz - sx*sy*cz
		w = cx*cy*cz + sx*sy*sz
	
		return quaternion([x, y, z, w])

	def multiply(q1, q2):
		x1, y1, z1, w1 = q1
		x2, y2, z2, w2 = q2
	
		w = w1*w2 - x1*x2 - y1*y2 - z1*z2
		x = w1*x2 + x1*w2 + y1*z2 - z1*y2
		y = w1*y2 + y1*w2 + z1*x2 - x1*z2
		z = w1*z2 + z1*w2 + x1*y2 - y1*x2
	
		return np.array([x, y, z, w])

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


class Transform(Node):
	def __init__(self, position, rotation, scale, **kwargs):
		super().__init__(**kwargs)
		self.position = position
		self.rotation = rotation
		self.scale = scale

	@property
	def dim(self): return len(self.position)
	
	@property
	def T(self): return Matrix.T(self.position)

	@property
	def R(self): pass

	@property
	def S(self): return Matrix.S(self.scale)

	def TRS(self):
		#return functools.reduce(lambda acc, x: x @ acc, [x.local_TRS() for x in [self] + self.ancestors()])
		return functools.reduce(lambda acc, x: acc @ x, [x.local_TRS() for x in reversed([self] + self.ancestors())])

	def inverse_TRS(self): return np.linalg.inv(self.TRS())

	def local_TRS(self):	#local transformation matrix
		#print(self.T, self.R, self.S)
		return self.T @ self.R @ self.S

	def local_inverse_TRS(self): #local inverse transformation matrix
		return np.linalg.inv(self.local_TRS())
		#return np.linalg.inv(self.S) @ np.linalg.inv(self.R) @ np.linalg.inv(self.T)

	def to_local(self, point):
		p = np.append(point, 1)
		return (self.inverse_TRS() @ p)[:self.dim]

	def to_global(self, point):
		p = np.append(point, 1)
		return (self.TRS() @ p)[:self.dim]

	@property
	def global_position(self): return self.to_global(self.position)

	@property
	def basis(self):	# BASIS (n×n matrix of world axes) -> upper-left n×n
		return self.TRS()[:self.dim, :self.dim]

class Node2D(Transform):	#Node):
	def __init__(self, position=np.zeros(2), rotation=0.0, scale=np.ones(2), **kwargs):
		super().__init__(position, rotation, scale, **kwargs)

	@property
	def R(self): return Matrix.R2(self.rotation)

class Node3D(Transform):	#Node):):
	def __init__(self, position=np.zeros(3), rotation=quaternion([0, 0, 0, 1]), scale=np.ones(3), **kwargs):
		super().__init__(position, rotation, scale, **kwargs)

	@property
	def euler(self): return self.rotation.to_euler()
	@euler.setter
	def euler(self, value): self.rotation = quaternion.from_euler(value)

	@property
	def R(self): return Matrix.R3(self.rotation)







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

