
import numpy as np
import pyx.generic.generic as generic


def fade(scene, obj, end, start_time, duration, start=None, ease=None):
	k = 'attrib/opacity'
	if start is None:
		generic.set(obj, k, generic.get(obj, k, 1.))	# Initialize opacity to avoid None
	scene.animate(target=obj, property=k, start=start, end=end, start_time=start_time, duration=duration, ease=ease)


def fade_in(scene, obj, start_time, duration, ease=None):	#alpha from 0 to 1
	generic.set(obj, 'attrib/opacity', generic.get(obj, k, 0.))
	return fade(scene, obj, start=0., end=1., start_time=start_time, duration=duration, ease=ease)


def fade_out(scene, obj, start_time, duration, ease=None):	#alpha from 1 to 0
	return fade(scene, obj, start=1., end=0., start_time=start_time, duration=duration, ease=ease)


def scale_from(scene, obj, point=None, start_time=0, duration=1, ease=None,
		scale_property='xform/scale', position_property='center'):

	scale = generic.get(obj, scale_property)
	center = generic.get(obj, position_property)

	animations = [
		scene.animate(
			target=obj,
			property=scale_property,
			start=np.zeros_like(scale),
			end=scale,
			start_time=start_time,
			duration=duration,
			ease=ease
		)
	]

	if point is not None:
		animations.append(
			scene.animate(
				target=obj,
				property=position_property,
				start=point,
				end=center,
				start_time=start_time,
				duration=duration,
				ease=ease
			)
		)

	return animations


def grow_from_center(scene, obj, start_time=0, duration=1, ease=None):
	return scale_from(scene, obj, start_time=start_time,
		duration=duration, ease=ease)


def grow_from_point(scene, obj, point, start_time=0, duration=1, ease=None):
	return scale_from(scene, obj, point, start_time, duration, ease)


def grow_from_edge(scene, obj, point, start_time=0, duration=1, ease=None):
	return grow_from_point(scene, obj, point, start_time, duration, ease)


def shrink_to_center(scene, obj, start_time=0, duration=1, ease=None,
		scale_property='xform/scale'):

	scale = generic.get(obj, scale_property)

	return scene.animate(
		target=obj,
		property=scale_property,
		start=scale,
		end=np.zeros_like(scale),
		start_time=start_time,
		duration=duration,
		ease=ease
	)


def spin_in_from_nothing(scene, obj, start_time=0, duration=1, ease=None,
		scale_property='xform/scale',
		rotation_property='xform/rotation'):

	scale = generic.get(obj, scale_property)
	rotation = generic.get(obj, rotation_property)

	return [
		scene.animate(
			target=obj,
			property=scale_property,
			start=np.zeros_like(scale),
			end=scale,
			start_time=start_time,
			duration=duration,
			ease=ease
		),
		scene.animate(
			target=obj,
			property=rotation_property,
			start=rotation - 2 * np.pi,
			end=rotation,
			start_time=start_time,
			duration=duration,
			ease=ease
		)
	]


def scale_in_place(scene, obj, factor=2, start_time=0, duration=1, ease=None,
		scale_property='xform/scale'):

	scale = generic.get(obj, scale_property)

	return scene.animate(
		target=obj,
		property=scale_property,
		start=scale,
		end=scale * factor,
		start_time=start_time,
		duration=duration,
		ease=ease
	)



def draw_curve(scene, obj, start_time=0, duration=1, ease=None):
	curve = copy.deepcopy(obj)
	group = geo.group([obj])
	scene.add(group)

	def update(t):
		#print(t)
		group[0] = curve.subcurve(0, t).set(**getattr(curve, 'attrib', {}))

	return scene.animate(
		target=group,
		property='geometry',
		start_time=start_time,
		duration=duration,
		ease=ease,
		value_function=lambda t: update(t)
	)
