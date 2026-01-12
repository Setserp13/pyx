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
		obj.sr = sr
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



"""def add_silence_start(input_path, output_path, seconds):
	y, sr = sf.read(input_path)
	out = np.concatenate([silence(seconds, sr, y.shape[1] if y.ndim > 1 else 1, y.dtype), y], axis=0)
	sf.write(output_path, out, sr)

def concat(audios, output_path="output.wav", gap_seconds=0.0):
	audio_parts = []
	sr = None
	
	for f in audios:
		data, samplerate = sf.read(f)
		if sr is None:
			sr = samplerate
		audio_parts.append(data)

		# add silence (except after last file)
		if gap_seconds > 0 and f != audios[-1]:
			audio_parts.append(silence(gap_seconds, sr, data.shape[1] if data.ndim > 1 else 1, data.dtype))

	out = np.concatenate(audio_parts, axis=0)
	sf.write(output_path, out, sr)
	#print("Saved:", output_path)"""

class soundwave():
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
	def duration(self): return librosa.get_duration(path=self.path)


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
