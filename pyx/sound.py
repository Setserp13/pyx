import numpy as np
import librosa
import soundfile as sf
from pyx.timeline import Interval, Time

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
		data = np.zeros(samples, dtype=dtype)
	else:
		data = np.zeros((samples, channels), dtype=dtype)

	return audio(data, sr=sr)

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

	"""def amp_at(self, t, channel=None):
		if not (0 <= t < self.duration):
			raise ValueError("Time outside audio duration")
	
		idx = t * self.sr
		i = int(idx)
		f = idx - i
	
		if i + 1 >= len(self):
			sample = self[i]
		else:
			sample = npx.lerp(self[i], self[i + 1], f)
	
		if channel is not None and self.ndim > 1:
			sample = sample[channel]
	
		return np.abs(sample)"""
		
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

	def to_mono(self):	# Raw mono waveform
		samples = self.astype(np.float32, copy=True)
	
		if samples.ndim > 1:
			samples = samples.mean(axis=1)
	
		return audio(samples, sr=self.sr)

	def frames(y, frame_length, hop_length, center=False):
		"""Slice an array into overlapping frames."""
	
		if frame_length <= 0 or hop_length <= 0:
			raise ValueError('frame_length and hop_length must be positive')
	
		y = np.asarray(y)
	
		if y.ndim not in (1, 2):
			raise ValueError('Expected a 1D or 2D array')
	
		if center:
			pad = frame_length // 2
			padding = [(pad, pad)] + [(0, 0)] * (y.ndim - 1)
			y = np.pad(y, padding)
	
		if len(y) < frame_length:
			return np.empty((0, frame_length) + y.shape[1:], dtype=y.dtype)
	
		n_frames = 1 + (len(y) - frame_length) // hop_length
	
		shape = (n_frames, frame_length) + y.shape[1:]
		strides = (y.strides[0] * hop_length,) + y.strides
	
		return np.lib.stride_tricks.as_strided(
			y,
			shape=shape,
			strides=strides,
			writeable=False
		)

	
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

	def stft(self, frame_length=2048, hop_length=512):
		w = np.hanning(frame_length).reshape(1, frame_length, 1)
	
		return self.apply(lambda f: np.fft.rfft(f * w, axis=1), frame_length, hop_length)

	def _fft_magnitude(frame, window):
		"""Calculate the normalized single-sided magnitude spectrum."""
		n_fft = len(window)
	
		magnitudes = np.abs(np.fft.rfft(frame * window))
	
		magnitudes /= max(window.sum(), 1e-12)
	
		if n_fft % 2 == 0:
			magnitudes[1:-1] *= 2
		else:
			magnitudes[1:] *= 2
	
		return magnitudes
	
	@staticmethod
	def frequencies(sr, n_fft):
		"""Return the frequency values (Hz) for each bin of a real FFT."""
		if sr <= 0:
			raise ValueError('sr must be positive')
		if n_fft <= 0:
			raise ValueError('n_fft must be positive')
	
		return np.fft.rfftfreq(n_fft, d=1 / sr)
	
	def spectrum(self, n_fft=2048):
		"""Return (frequencies, magnitudes) for an audio signal."""
		if n_fft <= 0:
			raise ValueError('n_fft must be positive')
	
		samples = self.to_mono()
	
		if len(samples) == 0:
			raise ValueError('Audio is empty')
	
		if len(samples) < n_fft:
			samples = np.pad(samples, (0, n_fft - len(samples)))
		else:
			samples = samples[:n_fft]
	
		window = np.hanning(n_fft)
		magnitudes = _fft_magnitude(samples, window)
	
		return audio.frequencies(self.sr, n_fft), magnitudes
	
	def spectrogram(self, n_fft=2048, hop_length=512, center=True):
		"""Return (times, frequencies, magnitudes) using STFT."""
		if n_fft <= 0 or hop_length <= 0:
			raise ValueError('n_fft and hop_length must be positive')
	
		samples = self.to_mono()
	
		if len(samples) == 0:
			raise ValueError('Audio is empty')
	
		window = np.hanning(n_fft)
	
		spectra = [
			_fft_magnitude(frame, window)
			for frame in self.frames(n_fft, hop_length, center)
		]
	
		times = np.arange(len(spectra)) * hop_length / self.sr
		frequencies = audio.frequencies(self.sr, n_fft)
		magnitudes = np.asarray(spectra).T
	
		return times, frequencies, magnitudes
	
	def envelope(audio, fps, spf=100):
		"""Return RMS amplitude envelope."""
		hop_length = max(1, round(audio.sr / (fps * spf)))
	
		return audio.to_mono().rms(hop_length=hop_length)


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














def overlay(audio1, audio2, start_time=0):
	assert audio1.sr == audio2.sr, 'Sample rates must match'

	start = int(start_time * audio1.sr)
	end = max(len(audio1), start + len(audio2))

	shape = (end,) + audio1.shape[1:]
	result = np.zeros(shape, dtype=np.result_type(audio1, audio2, np.float32))

	result[:len(audio1)] += audio1
	result[start:start + len(audio2)] += audio2

	return audio(result, sr=audio1.sr)

class audio_clip(Interval):
	def __init__(self, media, start=0, media_start=0, duration=None):
		if duration is None:
			duration = media.duration - media_start

		super().__init__(start, start + duration)

		self.media = media
		self.media_start = Time(media_start)

	@property
	def media_end(self):
		return Time(self.media_start + self.duration)

	@property
	def segment(self):
		return self.media.segment(self.media_start, self.media_end)

class audio_track:
	def __init__(self, sr=44100):
		self.sr = sr
		self.items = []

	def add(self, audio, start=0, media_start=0, duration=None):
		clip = audio_clip(audio, start, media_start, duration)
		self.items.append(clip)
		return clip

	def append(self, audio, media_start=0, duration=None):
		return self.add(
			audio,
			start=self.duration,
			media_start=media_start,
			duration=duration
		)

	@property
	def duration(self):
		return max((x.end for x in self.items), default=Time(0))

	@property
	def audio(self):
		result = silence(float(self.duration), sr=self.sr)

		for clip in self.items:
			result = overlay(result, clip.segment, start_time=clip.start)

		return result
