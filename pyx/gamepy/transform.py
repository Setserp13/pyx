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

	def R3(q):
		q = q / np.linalg.norm(q)
		q_w, q_x, q_y, q_z = q
		R = np.array([
			[1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
			[2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_x * q_w)],
			[2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x**2 + q_y**2)]
		])
		return R

	"""def R3(axis, theta): #rotation 3D
		axis = np.asarray(axis, dtype=float)
		axis /= np.linalg.norm(axis)  # normalize the axis
		x, y, z = axis
		c = np.cos(theta)
		s = np.sin(theta)
		C = 1 - c
		return np.array([
			[c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
			[y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
			[z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
		])"""

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
	
	def from_euler(euler):
		roll, pitch, yaw = euler
		# Convert angles to radians if they are in degrees
		# roll, pitch, yaw = np.radians([roll, pitch, yaw])
	    
		# Compute half angles
		half_roll = roll / 2
		half_pitch = pitch / 2
		half_yaw = yaw / 2
	
		# Compute the individual quaternions for each axis
		qw_roll = np.cos(half_roll)
		qx_roll = np.sin(half_roll)
	    
		qw_pitch = np.cos(half_pitch)
		qy_pitch = np.sin(half_pitch)
	    
		qw_yaw = np.cos(half_yaw)
		qz_yaw = np.sin(half_yaw)
	
		# Quaternion for roll (around X-axis)
		q_roll = np.array([qw_roll, qx_roll, 0, 0])
	
		# Quaternion for pitch (around Y-axis)
		q_pitch = np.array([qw_pitch, 0, qy_pitch, 0])
	
		# Quaternion for yaw (around Z-axis)
		q_yaw = np.array([qw_yaw, 0, 0, qz_yaw])
	
		# Multiply quaternions in the order Yaw -> Pitch -> Roll
		q = quaternion.multiply(quaternion.multiply(q_yaw, q_pitch), q_roll)
	    
		return q

	def multiply(q1, q2):
		w1, x1, y1, z1 = q1
		w2, x2, y2, z2 = q2
		w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
		x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
		y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
		z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
		return np.array([w, x, y, z])



class Node2(Node):
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
