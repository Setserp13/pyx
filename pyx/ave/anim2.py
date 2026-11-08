import pyx.svgx as svgx
import pyx.lxmlx as lxmlx
import os
import subprocess
import cairosvg
import pyx.numpyx_geo as geo
import numpy as np
from pyx.mat.easing import *
import copy
import pyx.numpyx as npx
import pyx.generic.generic as generic

class Animation:
	def __init__(self, target, property, start=None, end=None, start_time=0,
			duration=1, ease=None, value_function=None, update=None, finish=None):

		self.target = target
		self.property = property
		self.start = start
		self.end = end
		self.duration = duration
		self.start_time = start_time
		self.ease = ease or (lambda t: t)
		self.value_function = value_function
		self.update = generic.Event(update)
		self.finish = generic.Event(finish)
		self.finished = False

	@property
	def end_time(self): return self.start_time + self.duration
	
	def apply(self, time):
		if time < self.start_time:
			return

		if self.finished:
			return
		
		if self.start is None:
			self.start = generic.get(self.target, self.property)	#self.target.get_value(self.property)
		
		if self.duration == 0:
			t = 1. if time >= self.start_time else 0.
		else:
			t = np.clip((time - self.start_time) / self.duration, 0, 1)
		t = self.ease(t)

		if self.value_function:
			value = self.value_function(t)
		else:
			value = npx.lerp(self.start, self.end, t)

		self.update.invoke(t)
		
		generic.set(self.target, self.property, value)

		if time > self.end_time:
			self.finish.invoke()	#one-time callback
			self.finished = True


class Timeline:
	def __init__(self, fps=30, duration=None):
		self.fps = fps
		self._duration = duration	# None means the duration is calculated automatically from the animations.
		self.animations = []

	def add(self, animation):
		self.animations.append(animation)
		return animation

	@property
	def duration(self):
		if self._duration is not None:
			return self._duration

		if not self.animations:
			return 0

		return max(
			animation.start_time + animation.duration
			for animation in self.animations
		)

	@duration.setter
	def duration(self, value):
		self._duration = value

	@property
	def frame_count(self):
		return round(self.fps * self.duration)

	def apply(self, time):
		for animation in self.animations:
			animation.apply(time)


class Scene:
	def __init__(self, width=640, height=360, fps=30, duration=5):
		self.width = width
		self.height = height

		self.timeline = Timeline(fps, duration)
		self.objects = []

	@property
	def size(self): return np.array([self.width, self.height])

	@size.setter
	def size(self, value):
		self.width = value[0]
		self.height = value[1]

	def add(self, *objects):
		self.objects.extend(objects)
		return self

	def animate(self, *args, **kwargs):
		return self.timeline.add(
			Animation(*args, **kwargs)
		)

	def assign(self, target, property, value, time):
		"""Set a property to a value at a specific time."""
		self.animate(
			target=target,
			property=property,
			start=value,
			end=value,
			start_time=time,
			duration=0
		)
	
	def hold(self, target, property, value, start_time, duration):
		"""Keep a property at a value during a time interval."""
		self.animate(
			target=target,
			property=property,
			start=value,
			end=value,
			start_time=start_time,
			duration=duration
		)
	
	def sequence(self, target, property, start, animations, start_time=0):
		result = []

		for end, duration, *args in animations:
			ease = args[0] if args else None

			animation = self.animate(
				target=target,
				property=property,
				start=start,
				end=end,
				start_time=start_time,
				duration=duration,
				ease=ease
			)

			result.append(animation)

			# Continue from the previous animation's endpoint.
			start = end
			start_time += duration

		return result

	def follow_path(self, target, curve, property='center',
			start_time=0, duration=1, ease=None):

		def update(t):
			setattr(target, property, curve.get_point01(t))

		animation = self.animate(
			target=target,
			property=property,
			start=0,
			end=1,
			start_time=start_time,
			duration=duration,
			ease=ease
		)

		animation.value_function = update
		return animation

	def draw(self, time=0):
		self.timeline.apply(time)
		return canvas(self.objects, self.size)


class canvas(geo.group):
	def __init__(self, items, size):
		super().__init__(items)

		self.rect = geo.rect(np.zeros(2), size).set(fill='black', stroke='none')

		self.insert(0, self.rect)

	def draw(self):
		objects = copy.deepcopy(self)

		#for x in objects[1:]:
		#	x.global_aabb = self.rect.denormalize_rect(x.global_aabb)	# Use global_aabb to preserve the effects of parent and local transforms.
	
		return objects

	def save_svg(self, path='drawing.svg', overwrite=True):
		if not path.lower().endswith('.svg'):
			path += '.svg'

		if not overwrite:
			path = osx.to_distinct(path)

		result = svgx.svg(*self.rect.size)
		result.extend(x.to_svg() for x in self.draw())

		lxmlx.save(result, path)

		return path


def render_frames(scene, directory='frames'):
	os.makedirs(directory, exist_ok=True)

	timeline = scene.timeline

	for i in range(timeline.frame_count):
		time = i / timeline.fps

		frame = scene.draw(time)

		filename = os.path.join(
			directory,
			f'frame_{i:05d}'
		)
		#for x in frame: print(x.aabb)
		frame.save_svg(filename)

	return directory


def render_video(scene, output='animation.mp4', directory='frames'):
	render_frames(scene, directory)

	for filename in os.listdir(directory):
		if filename.endswith('.svg'):
			source = os.path.join(directory, filename)
			target = os.path.splitext(source)[0] + '.png'

			cairosvg.svg2png(
				url=source,
				write_to=target,
				output_width=scene.width,
				output_height=scene.height
			)

	subprocess.run([
		'ffmpeg',
		'-y',
		'-framerate', str(scene.timeline.fps),
		'-i', os.path.join(directory, 'frame_%05d.png'),
		'-c:v', 'libx264',
		'-pix_fmt', 'yuv420p',
		output
	], check=True)

	return output

# H.264 with yuv420p does not support transparency, so alpha channels from PNG frames are discarded.
# For transparent video, use a codec that supports alpha (e.g. ProRes 4444).
# Otherwise, composite frames over a solid background before encoding.




