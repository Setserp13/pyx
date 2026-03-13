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

	def frames(y, frame_length, hop_length):	#Slice a data array into (overlapping) frames.
		
		if y.ndim == 1:
			n_frames = 1 + (len(y) - frame_length) // hop_length
			shape = (n_frames, frame_length)
			strides = (y.strides[0] * hop_length, y.strides[0])
	
		else:
			n_samples, n_channels = y.shape
			n_frames = 1 + (n_samples - frame_length) // hop_length
	
			shape = (n_frames, frame_length, n_channels)
			strides = (y.strides[0] * hop_length, y.strides[0], y.strides[1])
	
		return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

	def apply(self, func, frame_length, hop_length):
		frames = self.frames(frame_length, hop_length)
		values = func(frames)
		return audio(values, self.sr // hop_length)

	def rms(self, frame_length=2048, hop_length=512):
		return self.apply(lambda frames: np.sqrt(np.mean(frames**2, axis=1)), frame_length, hop_length)
	
	def short_time_energy(self, frame_length=2048, hop_length=512):
		return self.apply(lambda frames: np.sum(frames**2, axis=1), frame_length, hop_length)

	def spectral_flux(self, frame_length=2048, hop_length=512):
		def func(frames):
			spectrum = np.abs(np.fft.rfft(frames, axis=1))
			diff = np.diff(spectrum, axis=0)
			flux = np.sum(diff**2, axis=1)
			if flux.ndim == 1:
				flux = np.concatenate(([0], flux))
			else:
				flux = np.vstack([np.zeros((1, flux.shape[1])), flux])
			return flux
		return self.apply(func, frame_length, hop_length)
	

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

#Time unit conversion
#time (seconds) ↔ samples ↔ frames

def frames_to_samples(frames, hop_length, n_fft=None):
	frames = np.asarray(frames)
	offset = 0
	if n_fft is not None:
		offset = n_fft // 2
	return frames * hop_length + offset

def samples_to_frames(samples, hop_length, n_fft=None):
	samples = np.asarray(samples)
	offset = 0
	if n_fft is not None:
		offset = n_fft // 2
	return ((samples - offset) / hop_length).astype(int)

def samples_to_time(samples, sr):
	samples = np.asarray(samples)
	return samples / sr

def time_to_samples(times, sr):
	times = np.asarray(times)
	return (times * sr).astype(int)

def frames_to_time(frames, sr, hop_length, n_fft=None):
	return samples_to_time(frames_to_samples(frames, hop_length, n_fft), sr)

def time_to_frames(times, sr, hop_length, n_fft=None):
	return samples_to_frames(time_to_samples(times, sr), hop_length, n_fft)

###


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
