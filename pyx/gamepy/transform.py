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

def euler_to_quaternion(roll, pitch, yaw):
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
	q = quaternion_multiply(quaternion_multiply(q_yaw, q_pitch), q_roll)
    
	return q

def quaternion_multiply(q1, q2):
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
	z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
	return np.array([w, x, y, z])

"""class Node3(Noe):
	def __init__(self, position=np.zeros(3), rotation=np.zeros(3), scale=np.ones(3)):
		super().__init__(None, None)
		self.position = position
		self.rotation = rotation
		self.scale = scale

	def euler_rotation_matrix(theta_x, theta_y, theta_z):
		R_x = rotation_x(theta_x)
		R_y = rotation_y(theta_y)
		R_z = rotation_z(theta_z)
		return np.dot(R_z, np.dot(R_y, R_x))

	def local_TRS(self):
		return T(self.position) @ R3(self.rotation) @ S(self.scale)

	def local_inverse_TRS(self):
		T_inv = T(-self.position)
		R_inv = R3(self.rotation).T  # Rotation inverse is the transpose
		S_inv = S(1/self.scale)
		return S_inv @ R_inv @ T_inv"""

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
