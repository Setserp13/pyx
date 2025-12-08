import math
import numpy as np
import pyx.numpyx as npx
import pyx.numpyx_geo as geo

def uv_sphere(radius=1.0, stacks=16, slices=32, center=np.zeros(3)):
	v = []
	uv = []
	for theta in npx.subdivide(0.0, math.pi, stacks):
		for phi in npx.subdivide(0.0, math.pi * 2.0, slices):
			v.append(npx.spherical_to_cartesian(radius, theta, phi) + center)
			uv.append(np.array([phi / (math.pi * 2), theta / math.pi]))
	return geo.Mesh(v, geo.enlongated_faces(*([slices] * stacks)), uv)
