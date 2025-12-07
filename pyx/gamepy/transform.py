import numpy as np
import pyx.numpyx as npx
from pyx.generic.node import Node
import functools

"""
Use R.T (transposta) — é mais rápido e numericamente mais estável que np.linalg.inv.
Se a matriz não for ortogonal (por exemplo contém escala, cisalhamento ou erro numérico), R.T não será a inversa — aí use np.linalg.inv.
Para rotações representadas por quaternions, a inversa (rotação oposta) é o conjugado do quaternion normalizado; ao converter para matriz, a transposta continua sendo a inversa.
"""

class Matrix:
	def S(vector): #scaling
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
		R4 = np.eye(4)
		R4[:3, :3] = R
		return R4

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

