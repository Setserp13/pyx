from pyx.mat.ease import Ease
from pyx.mat.mat import *
import cv2
import math

class Clip:
	def __init__(self, start_time, duration, function):
		self.start_time = start_time
		self.duration = duration
		self.function = function

	@property
	def end_time(self): return self.start_time + self.duration

	@end_time.setter
	def end_time(self, value): self.duration = value - self.start_time

	def update(self, t, *args):#, **kwargs):
		if self.start_time <= t <= self.end_time:
			self.function(t - self.start_time, *args)

class ClipGroup(Clip):
	def __init__(self, items=None):
		self.start_time = float('inf')
		self.end_time = float('-inf')
		self.items = []
		if items is not None:
			self.extend(items)
	
	def append(self, clip):
		self.items.append(clip)
		self.start_time = min(self.start_time, clip.start_time)
		self.end_time = max(self.end_time, clip.end_time)
	
	def extend(self, clips):
		for x in clips:
			self.append(x)

	def __getitem__(self, index): return self.items[index]

	def __setitem__(self, index, value): self.items[index] = value

	def __len__(self): return len(self.items)

	def update(self, t, *args):
		for x in self.items:
			x.update(t, *args)

def animate_property(object, property, start_time, duration, function, ease = Ease.QuadInOut):
	return Clip(start_time, duration, lambda t, *args: setattr(object, property, function(ease(t / duration))))

def tween_property(object, property, start_time, duration, start_value, end_value, ease = Ease.QuadInOut):
	return animate_property(object, property, start_time, duration, lambda t: lerp(start_value, end_value, t), ease)

def path_property(object, property, start_time, durations, start_value, values):#, eases):
	result = ClipGroup()
	for i in range(len(durations)):
		result.append(tween_property(object, property, start_time, durations[i], start_value, values[i]))
		start_time += durations[i]
		start_value = values[i]
	return result






class DictClip(Clip):
	def __init__(self, start_time, duration, make, clips=None, **init): # clips is a list of tuples of property name and clip, each clip in clips has func that receives prop value and t as input and returns a given property value
		self.start_time = start_time
		self.duration = duration
		self.make = make
		self.init = init
		self.clips = [] if clips == None else clips
		self.function = lambda t, *args: self.cfunc(t, *args)
		#print(self.clips, init)
		self.props = self.init

	#SET PROP FUNC
	def set_prop(self, name, start_time, duration, func):
		#clip = Clip(start_time, duration, func)
		def f(t, *args): self.props[name] = func(t)
		clip = Clip(start_time, duration, f)
		self.start_time = min(self.start_time, start_time)
		self.end_time = max(self.end_time, start_time + duration)
		self.clips.append([name, clip])

	def set_prop01(self, name, start_time, duration, func, ease = Ease.QuadInOut):
		self.set_prop(name, start_time, duration, lambda t: func(ease(t / duration)))


	def tween(self, name, start_time, duration, start_value, end_value, ease = Ease.QuadInOut):
		self.set_prop01(name, start_time, duration, lambda  t: lerp(start_value, end_value, t), ease)


	def loop(self, name, start_time, duration, n, start_value, end_value, ease = Ease.QuadInOut, length_based=False):
		if length_based:
			n = duration / n
		length = 1 / n
		self.set_prop01(name, start_time, duration, lambda  t: lerp(start_value, end_value, n * ping_pong(t, length)), ease)


	def keep(self, name, start_time, duration, value):
		self.set_prop(name, start_time, duration, lambda  t: value)


	def path(self, name, start_time, durations, start_value, values):#, eases):
		for i in range(len(durations)):
			self.tween(name, start_time, durations[i], start_value, values[i])
			start_time += durations[i]
			start_value = values[i]

	def cfunc(self, t, *args): # composite function
		for x in self.clips:
			x[1].update(t + self.start_time)
			#self.props[x[0]] = x[1].update(self.props[x[0]], t + self.start_time)
		args[0][:] = self.make(args[0], **self.props)[:]
		#return self.make(obj, **self.props)


	def fadein(self, start_time, duration):
		self.set_prop01('color', start_time, duration, lambda x, t: (x[0], x[1], x[2], int(t * 255)))

	def fadeout(self, start_time, duration):
		self.set_prop01('color', start_time, duration, lambda x, t: (x[0], x[1], x[2], int((1 - t) * 255)))

def progressbar(i, total, size=50):
	progress = i / (total - 1)
	fill = math.floor(progress * size)
	bar = '#' * fill + '_' * (size - fill)
	print(f'{bar}{100*progress:.2f}% {i+1}/{total}', end='\r')	

def generateVideo(fps, width, height, frame_count, *clips, filename='output.mp4'):
	# Create video writer object
	#fourcc = cv2.VideoWriter_fourcc(*'H264')
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#print(clips)
	out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)
	out.set(cv2.CAP_PROP_BITRATE, 10000)
	img = np.zeros((height, width, 4), dtype=np.uint8)
	for i in range(0, frame_count):
		#print(str(i) + '/' + str(frame_count))
		#img = np.zeros((height, width, 4), dtype=np.uint8)
		#img[:, :, 3] = 255
		img[:,:] = (0,0,0,255)
		t = i / fps
		shape = img.shape
		
		for x in clips:
			#img = x.update(img, t)
			x.update(t, img)

		if img.shape != shape: #PRESTA ATENÇÃO NISTO, POIS SE ESTE ERRO OCORRER, O FRAME NÃO SERÁ ADICIONADO NO VÍDEO E HAVERÁ O ADIANTAMENTO DAS PARTES POSTERIORES
			print([img.shape, shape])
		out.write(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
		progressbar(i, frame_count)
	out.release()
	return out







import imageio
import numpy as np
import os
import cv2
from tqdm import tqdm
import subprocess

def generateVideoWithAlpha(fps, width, height, frame_count, *clips, filename='output.mov'):
	# Abre o writer com codec PNG e RGBA
	writer = imageio.get_writer(
		filename,
		format='FFMPEG',
		mode='I',
		fps=fps,
		codec='png',
		output_params=['-pix_fmt', 'rgba']
	)

	#tmp_folder = "__temp_frames__"
	#os.makedirs(tmp_folder, exist_ok=True)

	print("Gerando frames com transparência...")
	for i in tqdm(range(frame_count)):
		t = i / fps
		img = np.zeros((height, width, 4), dtype=np.uint8)
		img[:, :] = (0, 0, 0, 0)  # fundo transparente	#img[:, :, :] = (0, 0, 0, 0)

		for clip in clips:
			clip.update(t, img)  # clip deve desenhar em RGBA

		#frame_path = os.path.join(tmp_folder, f"frame_{i:05d}.png")
		#imageio.imwrite(frame_path, img)
		writer.append_data(img)

	writer.close()
	print("Codificando vídeo final com canal alfa...")

	"""for clip in clips:
		for i in tqdm(range(int(clip.start_time * fps), int(clip.end_time * fps))):
			frame_path = os.path.join(tmp_folder, f"frame_{i:05d}.png")
			frame = imageio.imread(frame_path)
			t = i / fps
			clip.update(t, frame)
			imageio.imwrite(frame_path, frame)""" #IN CASE YOU WANNA WRITE CLIP BY CLIP

	# Escreve os frames
	"""for i in tqdm(range(frame_count)):
		frame_path = os.path.join(tmp_folder, f"frame_{i:05d}.png")
		frame = imageio.imread(frame_path)
		writer.append_data(frame)
	writer.close()

	# Limpa os frames temporários
	for f in os.listdir(tmp_folder):
		os.remove(os.path.join(tmp_folder, f))
	os.rmdir(tmp_folder)"""

	print("Vídeo com transparência salvo como:", filename)

def add_audio_to_video(video_path, audio_path, output_path):
	subprocess.run([
		"ffmpeg",
		"-y",                      # overwrite if exists
		"-i", video_path,
		"-i", audio_path,
		"-c:v", "copy",            # don't re-encode video
		"-c:a", "aac",             # encode audio to AAC
		"-shortest",               # stop when shortest stream ends
		output_path
	])
