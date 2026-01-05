import numpy as np
import librosa
import soundfile as sf

def add_silence_start(input_path, output_path, seconds):
	y, sr = sf.read(input_path)
	n_samples = int(seconds * sr)
	if y.ndim == 1:
		silence = np.zeros(n_samples, dtype=y.dtype)
	else:
		silence = np.zeros((n_samples, y.shape[1]), dtype=y.dtype)
	out = np.concatenate([silence, y], axis=0)
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
			silence = np.zeros(int(sr * gap_seconds))
			# if stereo, match channels
			if data.ndim > 1:
				silence = np.zeros((int(sr * gap_seconds), data.shape[1]))
			audio_parts.append(silence)

	out = np.concatenate(audio_parts, axis=0)
	sf.write(output_path, out, sr)
	#print("Saved:", output_path)

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
