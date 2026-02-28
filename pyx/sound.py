import numpy as np
import librosa
import soundfile as sf

def noise(n):
	return np.random.randn(n)

def normalize(x):
	m = np.max(np.abs(x))
	return x if m == 0 else x / m

def silence(duration, sr=22050, channels=1, dtype=np.float32):
	"""
	Return a silent audio segment.

	duration : float  (seconds)
	sr       : int    (sample rate)
	channels : int    (1=mono, 2=stereo)
	dtype    : numpy dtype
	"""
	samples = int(duration * sr)

	if channels == 1:
		return np.zeros(samples, dtype=dtype)
	else:
		return np.zeros((samples, channels), dtype=dtype)

class audio(np.ndarray):
	def __new__(cls, input_array, sr=44100):
		obj = np.asarray(input_array).view(cls)
		obj.sr = sr	#Sample rate. How many audio samples per second
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.sr = getattr(obj, "sr", 44100)

	@classmethod
	def read(cls, path):
		data, samplerate = sf.read(path)
		return cls(data, sr=samplerate)

	def write(self, path):
		sf.write(path, self, self.sr)

	@property
	def duration(self): return len(self) / self.sr

	@property
	def channels(self): return self.shape[1] if self.ndim > 1 else 1

	def sample_at(self, t, channel=None):	#Return sample value at time t (seconds).
		if not (0 <= t < self.duration):
			raise ValueError("Time outside audio duration")
		idx = int(t * self.sr)
		sample = self[idx]
		if channel is not None and self.ndim > 1:
			sample = sample[channel]
		return sample

	def amp_at(self, t, channel=None):	#Return sample amplitude at time t (seconds).
		return np.abs(self.sample_at(t, channel))

	def segment(self, t_start, t_end):
		"""
		Return an audio segment between t_start and t_end (seconds).
		Works for mono and multi-channel audio.
		"""
		# Clamp times
		t_start = max(0.0, t_start)
		t_end = min(self.duration, t_end)

		if t_start >= t_end:
			# Return empty audio segment
			return audio(np.zeros((0,) + self.shape[1:], dtype=self.dtype), sr=self.sr)

		# Convert time to sample indices
		start_sample = int(t_start * self.sr)
		end_sample = int(t_end * self.sr)

		end_sample = min(end_sample, len(self))	# Safety clamp

		return audio(self[start_sample:end_sample], sr=self.sr)
	
	def normalize(self):	#Scale audio so peak amplitude becomes 1 (range -1 to 1).
		peak = np.max(np.abs(self))
		if peak == 0:
			return audio(self.copy(), sr=self.sr)
		return audio(self / peak, sr=self.sr)

	def to_mono(self):
		samples = self.copy()
		if samples.ndim > 1:	# Convert stereo → mono if needed
			samples = samples.mean(axis=1)
		return audio(samples, sr=self.sr)

	def to_rms(self, frame_length=2048, hop_length=512):
		y = self.to_mono()	# convert to mono if needed
		rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
		rms_sr = self.sr / hop_length
		return audio(rms, rms_sr)


def concatenate(ls, gap_seconds=0.0):
	arrays = []
	sr = None
	
	for x in ls:
		if sr is None:
			sr = x.sr
		arrays.append(x)

		# add silence (except after last file)
		if gap_seconds > 0 and x != ls[-1]:
			arrays.append(silence(gap_seconds, sr, x.channels, data.dtype))

	return audio(np.concatenate(arrays, axis=0), sr=sr)


#REMOVE THIS CLASS LATER, KEEP ALL INTO audio CLASS
"""class soundwave():
	def __init__(self, path):
		# --- Audio analysis ---
		self.path = path
		print(self.path)
		y, sr = librosa.load(self.path)
		frame_size = 2048
		hop = 512
		self.rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop)[0]
		self.rms = self.rms / np.max(self.rms)   # normalize 0–1

		self.times = librosa.frames_to_time(np.arange(len(self.rms)), sr=sr, hop_length=hop, n_fft=frame_size)

	def amp_at(self, t):
		for i in range(len(self.times) - 1):
			if self.times[i] <= t < self.times[i+1]:
				return self.rms[i]
		return self.rms[-1]

	@property
	def duration(self): return librosa.get_duration(path=self.path)"""


def sine_wave(freq, t, phase=0.0):	#pure tone
	return np.sin(2 * np.pi * freq * t + phase)

def square_wave(freq, t, phase=0.0):
	return np.sign(np.sin(2 * np.pi * freq * t + phase))

def triangle_wave(freq, t, phase=0.0):
	return 2 * np.arcsin(np.sin(2 * np.pi * freq * t + phase)) / np.pi

def saw_wave(freq, t, phase=0.0):	#sawtooth wave
	phase_cycles = phase / (2 * np.pi)
	x = t * freq + phase_cycles
	return 2 * (x - np.floor(0.5 + x))

def periodic_wave(
	frequency,
	duration,
	sr=44100,
	volume=0.5,
	phase=0.0,
	waveform=sine_wave,
	channels=1,
	dtype=np.float32
):
	"""
	Generate a pure tone.

	frequency : float   (Hz)
	duration  : float   (seconds)
	sr        : int     (sample rate)
	volume    : float   (0.0 – 1.0)
	phase     : float   (radians)
	wave      : str     ("sine", "square", "triangle", "saw")
	channels  : int     (1=mono, 2=stereo)
	dtype     : numpy dtype
	"""

	t = np.linspace(0, duration, int(sr * duration), endpoint=False)

	signal = waveform(frequency, t, phase)
	
	signal *= volume

	if channels > 1:
		signal = np.repeat(signal[:, None], channels, axis=1)

	return audio(signal.astype(dtype), sr=sr)
