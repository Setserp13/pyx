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
from pathlib import Path

import imageio.v2 as imageio
from tqdm import tqdm


class Animation:
	def __init__(self, start_time=0, duration=1):
		self.start_time = start_time
		self.duration = duration
		self.finish = generic.Event()
		self.finished = False

	@property
	def end_time(self): return self.start_time + self.duration

	def on_finish(self, method):
		self.finish.append(method)
		return self

	def _apply(self, time): raise NotImplementedError
		
	def apply(self, time):	#0 <= t <= duration
		if time < self.start_time or self.finished:
			return

		self._apply(time)

		if time >= self.end_time:
			self.finish.invoke()	#one-time callback
			self.finished = True

class FunctionTweener(Animation):
	def __init__(self, function, start_time=0, duration=1):
		super().__init__(start_time, duration)
		self.function = generic.Event([function])

	def _apply(self, time):
		t = np.clip(time - self.start_time, 0, self.duration)
		self.function.invoke(t)

class PropertyTweener(Animation):
	def __init__(self, target, property, start=None, end=None, start_time=0, duration=1, ease=None, value_function=None):
		super().__init__(start_time, duration)
		self.target = target
		self.property = property
		self.start = start
		self.end = end
		self.ease = ease or (lambda t: t)
		self.value_function = value_function
		self.update = generic.Event()

	def on_update(self, method):
		self.update.append(method)
		return self

	def _apply(self, time):
		
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

	@property
	def dt(self): return 1. / self.fps	#delta time

class Scene:
	def __init__(self, size=[640, 360], fps=30, duration=None, transparent=False):
		self.size = np.array(size)

		self.timeline = Timeline(fps, duration)
		self.objects = []

		self.transparent = transparent

	@property
	def channels(self): return 4 if transparent else 3
	
	@property
	def rect(self): return geo.rect(np.zeros(2), self.size)

	@property
	def width(self): return self.size[0]
	@property
	def height(self): return self.size[1]
	@width.setter
	def width(self, value):
		self.size[0] = value
	@height.setter
	def height(self, value):
		self.size[1] = value

	def add(self, *objects):
		self.objects.extend(objects)
		return self

	def animate(self, *args, **kwargs):
		return self.timeline.add(
			PropertyTweener(*args, **kwargs)
		)

	def animate_function(self, *args, **kwargs):
		return self.timeline.add(
			FunctionTweener(*args, **kwargs)
		)
	
	def assign(self, target, property, value, time):
		"""Set a property to a value at a specific time."""

		def update(t): return value

		animation = self.animate(
			target=target,
			property=property,
			start=value,
			end=value,
			start_time=time,
			duration=0
		)

		animation.value_function = update
		return animation
	
	def hold(self, target, property, value, start_time, duration):
		"""Keep a property at a value during a time interval."""

		def update(t): return value

		animation = self.animate(
			target=target,
			property=property,
			start=value,
			end=value,
			start_time=start_time,
			duration=duration
		)

		animation.value_function = update
		return animation
	
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
		return geo.group(self.objects)

	def iter_frames(scene):
		for i in tqdm(range(scene.timeline.frame_count), desc='Iterating frames'):
			time = i / scene.timeline.fps
			yield i, scene.draw(time)
	
	def render_frames(scene, output_dir='frames', prefix='frame_', digits=5, format='png'):
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
	
		format = format.lower()
	
		if format not in ('png', 'svg'):
			raise ValueError(f'Unsupported frame format: {format}')
	
		print(f'Generating {format.upper()} frames...')
	
		try:
			for i, frame in scene.iter_frames():
				filename = f'{prefix}{i:0{digits}d}.{format}'
				filepath = str(output_dir / filename)
	
				if format == 'png':
					img = np.zeros((scene.height, scene.width, scene.channels), dtype=np.uint8)
					frame.draw(img)
					imageio.imwrite(filepath, img)
	
				elif format == 'svg':
					frame.save_svg(filepath, size=scene.size, fit_to_content=False)
	
		finally:
			print('Frame rendering complete.')
	
		return output_dir

	"""def render_frames(scene, output_dir='frames', transparent=False, prefix='frame_', digits=5):
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
	
		channels = 4 if transparent else 3
	
		print('Generating frames...')
	
		try:
			for i, frame in scene.iter_frames():
				img = np.zeros((scene.height, scene.width, channels), dtype=np.uint8)
				frame.draw(img)
	
				filename = f'{prefix}{i:0{digits}d}.png'
				imageio.imwrite(str(output_dir / filename), img)
	
		finally:
			print('Frame rendering complete.')
	
		return output_dir

	def render_frames_svg(scene, output_dir='frames', prefix='frame_', digits=5):
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
	
		print('Generating SVG frames...')
	
		try:
			for i, frame in scene.iter_frames():
				filename = f'{prefix}{i:0{digits}d}.svg'
				frame.save_svg(str(output_dir / filename), size=scene.size, fit_to_content=False)
	
		finally:
			print('SVG frame generation complete.')
	
		return output_dir"""
	
	def render_video(scene, output='animation.mp4', transparent=False, codec=None):
		if output is None:
			output = 'animation.mov' if transparent else 'animation.mp4'
	
		codec = codec or ('png' if transparent else 'libx264')
	
		output_params = (
			['-pix_fmt', 'rgba']
			if transparent else
			['-pix_fmt', 'yuv420p']
		)
	
		writer = imageio.get_writer(
			output,
			format='FFMPEG',
			mode='I',
			fps=scene.timeline.fps,
			codec=codec,
			output_params=output_params,
		)
	
		channels = 4 if transparent else 3
	
		print('Generating video...')
	
		try:
			for i, frame in scene.iter_frames():
				img = np.zeros((scene.height, scene.width, channels), dtype=np.uint8)
				frame.draw(img)
				writer.append_data(img)
	
		finally:
			writer.close()
	
		print('Video encoding complete.')
	
		return output




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




