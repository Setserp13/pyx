import pyx.numpyx as npx
import pyx.numpyx_geo as geo

def circle_circle_collision(c1, c2):
	d = geo.circle_circle_delta(c1, c2)
	if d > 0:
		return None
	normal = npx.normalize(c2.center - c1.center)
	point = c1.center + normal * c1.radius
	return { "normal": normal, "penetration": -d, "point": point }

def circle_line_collision(c, l):
	d = geo.circle_line_delta(c, l)
	if d > 0:
		return None
	normal = npx.normalize(project_on_line(c.center, l) - c.center)
	point = c.center + normal * c.radius
	return { "normal": normal, "penetration": -d, "point": point }	

def circle_segment_collision(c, l):
	d = geo.circle_segment_delta(c, l)
	if d > 0:
		return None
	p = geo.point_segment_closest_point(c.center, l)
	normal = npx.normalize(p - c.center)
	point = c.center + normal * c.radius
	return { "normal": normal, "penetration": -d, "point": point }	

def circle_polyline_collision(c, v):
	hits = [circle_segment_collision(c, x) for x in v.edges()]
	hits.sort(key=lambda x: 0. if x is None else x['penetration'], reverse=True)
	return hits[0]

def circle_rect_collision(c, r):
	return circle_polyline_collision(c, npx.rect2.corners(r))




COLLISION_TABLE = {
	(geo.circle, geo.circle): circle_circle_collision,
	(geo.circle, geo.line): circle_line_collision,
	(geo.circle, geo.segment): circle_segment_collision,
	(geo.circle, geo.polyline): circle_polyline_collision,
	(geo.circle, npx.rect): circle_rect_collision,
}

def collision(a, b):
	key = (type(a), type(b))
	func = COLLISION_TABLE.get(key)

	if func:
		return func(a, b)

	# try reversed order
	key = (type(b), type(a))
	func = COLLISION_TABLE.get(key)

	if func:
		res = func(b, a)
		if res is None:
			return None
		return {**res, "normal": -res["normal"]}

	raise NotImplementedError((type(a), type(b)))
