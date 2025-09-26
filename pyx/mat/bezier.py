import math
import numpy as np
import pyx.numpyx as npx

class bezier(np.ndarray):
	def __new__(cls, input_array, endpoints=None):
		# Convert input to ndarray and view it as MyArray
		obj = np.asarray(input_array).view(cls)
		#obj.endpoints = [True] * len(input_array) if endpoints is None else endpoints
		obj.endpoints = list(range(len(input_array))) if endpoints is None else endpoints
		return obj

	def __array_finalize__(self, obj):
		# Called when the object is created via view/slicing
		if obj is None: return
		# You can set custom attributes here if needed
		self.my_attribute = getattr(obj, 'my_attribute', 'default')

	def path(self):
		result = []
		result.append(self[self.endpoints[0]:self.endpoints[0]+1])
		for i in range(len(self.endpoints) - 1):
			result.append(self[self.endpoints[i]+1:self.endpoints[i+1]+1])
		result.append(np.concatenate([self[self.endpoints[-1]+1:], self[:self.endpoints[0]+1]]))
		return result

	def d(self):
		d = ''
		for x in self.path():
			d += 'MQC'[len(x) - 1]
			d += ' ' + ' '.join([str(y) for y in x.flatten()]) + ' '
		#print(d)
		return d

def smooth_control_points(p, index, extents=10):
	angle = p.vertex_angle(index)
	rot = (math.pi - angle.size()) * 0.5
	cp = npx.normalize(npx.rotate(angle.vectors()[0], rot)) * extents
	cp2 = npx.normalize(npx.rotate(angle.vectors()[1], -rot)) * extents	
	return [cp, cp2]

def smooth_polyline(p, extents=10, closed=True):
	points = []
	endpoints = []
	cps = [smooth_control_points(p, i, extents=extents) for i in range(len(p))]
	for i in range(len(p) - (0 if closed else 1)):
		j = (i + 1) % len(p)
		points += [p[i], cps[i][1] + p[i], cps[j][0] + p[j]]
		endpoints += [3 * i]
	if not closed:
		points.append(p[-1])
		endpoints.append(3 * (len(p) - 1))
	return bezier(points, endpoints)
