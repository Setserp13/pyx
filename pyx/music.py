import numpy as np
import soundfile as sf
from pyx.sound import normalize

#12 equal temperament	#In modern times, 12-ET is usually tuned relative to a standard pitch of 440 Hz, called A440, meaning one note, A4 (the A in the 4th octave of a typical 88-key piano)
def ET12(n, A4_FREQ = 440.0): return A4_FREQ * (2 ** (n / 12))

def chromatic_scale(A4_FREQ = 440.0): #Returns frequencies of A A# B C C# D D# E F F# G G#
	return [ET12(i, A4_FREQ) for i in range(12)]

def kick(duration=0.5, freq_start=150, freq_end=40, amp=0.9, sr=44100):
	t = np.linspace(0, duration, int(sr * duration), False)

	# Exponential pitch drop
	freq = freq_start * (freq_end / freq_start) ** (t / duration)
	phase = 2 * np.pi * np.cumsum(freq) / sr

	# Amplitude envelope
	env = np.exp(-t * 8)

	signal = np.sin(phase) * env * amp
	return normalize(signal)

def snare(duration=0.3, tone_freq=180, noise_amp=0.7, tone_amp=0.3, sr=44100):
	t = np.linspace(0, duration, int(sr * duration), False)

	# Noise burst
	noise = np.random.randn(len(t)) * np.exp(-t * 12)

	# Body tone
	tone = np.sin(2 * np.pi * tone_freq * t) * np.exp(-t * 20)

	signal = noise * noise_amp + tone * tone_amp
	return normalize(signal)

def hihat(duration=0.08, amp=0.6, sr=44100):
	t = np.linspace(0, duration, int(sr * duration), False)

	# White noise
	noise = np.random.randn(len(t))

	# Fast decay
	env = np.exp(-t * 60)

	# Simple highpass effect
	hihat = np.diff(noise, prepend=0)

	signal = hihat * env * amp
	return normalize(signal)
