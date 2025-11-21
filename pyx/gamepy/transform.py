import numpy as np
import pyx.numpyx as npx
from pyx.generic.node import Node
import functools

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

	def to_euler(q):
		"""
		Convert quaternion (x, y, z, w) to Euler angles (pitch, yaw, roll)
		using intrinsic XYZ rotation order.
		Returned angles are in radians.
		"""
		x, y, z, w = q
		
		# --- X (pitch) ---
		sinp = 2 * (w*x + y*z)
		cosp = 1 - 2 * (x*x + y*y)
		pitch = np.arctan2(sinp, cosp)
		
		# --- Y (yaw) ---
		siny = 2 * (w*y - z*x)
		siny = np.clip(siny, -1, 1)  # avoid domain errors
		yaw = np.arcsin(siny)
		
		# --- Z (roll) ---
		sinr = 2 * (w*z + x*y)
		cosr = 1 - 2 * (y*y + z*z)
		roll = np.arctan2(sinr, cosr)
		
		return np.array([pitch, yaw, roll])

	@staticmethod
	def from_euler(euler):
		roll, pitch, yaw = euler
	
		hr = roll  * 0.5
		hp = pitch * 0.5
		hy = yaw   * 0.5
	
		sr = np.sin(hr)
		cr = np.cos(hr)
		sp = np.sin(hp)
		cp = np.cos(hp)
		sy = np.sin(hy)
		cy = np.cos(hy)
	
		# Combine into quaternion (x,y,z,w)
		x = sr*cp*cy - cr*sp*sy
		y = cr*sp*cy + sr*cp*sy
		z = cr*cp*sy - sr*sp*cy
		w = cr*cp*cy + sr*sp*sy
	
		return quaternion([x, y, z, w])

	def multiply(q1, q2):
		x1, y1, z1, w1 = q1
		x2, y2, z2, w2 = q2
	
		w = w1*w2 - x1*x2 - y1*y2 - z1*z2
		x = w1*x2 + x1*w2 + y1*z2 - z1*y2
		y = w1*y2 + y1*w2 + z1*x2 - x1*z2
		z = w1*z2 + z1*w2 + x1*y2 - y1*x2
	
		return np.array([x, y, z, w])



class Node2D(Node):
	def __init__(self, position=np.zeros(2), rotation=0.0, scale=np.ones(2)):
		super().__init__(None, None)
		self.position = position
		self.rotation = rotation
		self.scale = scale

	@property
	def global_position(self):
		#print(self.TRS())
		return (self.TRS() @ np.append(self.position, 1))[:2]

	def TRS(self):
		#print(self.local_TRS())
		return functools.reduce(lambda acc, x: acc @ x, [x.local_TRS() for x in [self] + self.ancestors()])

	def inverse_TRS(self):
		return functools.reduce(lambda acc, x: acc @ x, [x.local_inverse_TRS() for x in [self] + self.ancestors()])

	def local_TRS(self): #local transformation matrix
		T, R, S = Matrix.T(self.position), Matrix.R2(self.rotation), Matrix.S(self.scale)
		#print(T, R, S)
		return T @ R @ S

	def local_inverse_TRS(self): #local inverse transformation matrix
		T_inv = Matrix.T(-self.position)
		R_inv = Matrix.R2(self.rotation).T  # Rotation inverse is the transpose
		S_inv = Matrix.S(1/self.scale)
		return S_inv @ R_inv @ T_inv

	def to_local(self, point):
		# ponto global vira homogêneo
		p = np.append(point, 1)

		# aplica a matriz TRS inversa acumulada (self + ancestrais)
		local = self.inverse_TRS() @ p

		return local[:2]

	def to_global(self, point):
		p = np.append(point, 1)
		return (self.TRS() @ p)[:2]




class Node3D(Node):
	def __init__(self, position=np.zeros(3), rotation=quaternion([0, 0, 0, 1]), scale=np.ones(3)):
		super().__init__(None, None)
		self.position = position       # vec3
		self.rotation = rotation       # quaternion (x,y,z,w)
		self.scale = scale             # vec3

	@property
	def euler(self): return self.rotation.to_euler()
	@euler.setter
	def euler(self, value): self.rotation = quaternion.from_euler(value)
	
	@property
	def global_position(self):
		return (self.TRS() @ np.append(self.position, 1))[:3]

	# =====================================================================
	# COMBINED TRANSFORM (self + ancestors)
	# =====================================================================
	def TRS(self):
		return functools.reduce(
			lambda acc, x: acc @ x,
			[x.local_TRS() for x in [self] + self.ancestors()]
		)

	def inverse_TRS(self):
		return functools.reduce(
			lambda acc, x: acc @ x,
			[x.local_inverse_TRS() for x in [self] + self.ancestors()]
		)

	# =====================================================================
	# LOCAL TRANSFORM
	# =====================================================================
	def local_TRS(self):
		T = Matrix.T(self.position)
		R = Matrix.R3(self.rotation)    # rotation from quaternion
		S = Matrix.S(self.scale)
		return T @ R @ S

	def local_inverse_TRS(self):
		T_inv = Matrix.T(-self.position)
		
		# quaternion inverse (x,y,z,-w) normalized
		qx, qy, qz, qw = self.rotation
		q_inv = np.array([-qx, -qy, -qz, qw])
		
		R_inv = Matrix.R3(q_inv)
		S_inv = Matrix.S(1 / self.scale)
		
		return S_inv @ R_inv @ T_inv

	# =====================================================================
	# SPACE CONVERSION
	# =====================================================================
	def to_local(self, point):
		p = np.append(point, 1)
		return (self.inverse_TRS() @ p)[:3]

	def to_global(self, point):
		p = np.append(point, 1)
		return (self.TRS() @ p)[:3]

	# =====================================================================
	# BASIS (3×3 matrix of world axes)
	# =====================================================================
	@property
	def basis(self):
		M = self.TRS()[:3, :3]  # upper-left 3×3
		return M
