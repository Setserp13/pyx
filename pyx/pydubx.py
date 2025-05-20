import functools
from pydub import AudioSegment, silence
import pyx.rex as rex

#AudioSegment.merge = lambda ls: functools.reduce(lambda x, y: x+y, ls)#, AudioSegment.silent(0))
def concat(ls): return functools.reduce(lambda x, y: x+y, ls)#, AudioSegment.silent(0))

def overlay(ls): return functools.reduce(lambda a, b: a.overlay(b) if len(a) >= len(b) else b.overlay(a), ls)

def open_all(ls): return [AudioSegment.from_file(x) for x in ls]

def remove_silence(audio, silence_threshold=-60, min_silence_len=50, margin=0):
    return concat(split_on_silence(audio, silence_threshold=silence_threshold, min_silence_len=min_silence_len, margin=margin))

def split(audio, label_track): return [audio[int(x[0] * 1000):int(x[1] * 1000)] for x in rex.to_label_track(label_track)]

def strip_audio(audio, silence_threshold=-60, min_silence_len=50, margin=0):#-40):
    silent_ranges = silence.detect_silence(audio, silence_thresh=silence_threshold, min_silence_len=min_silence_len)
    #print(silent_ranges)
    if silent_ranges:
        start_trim = silent_ranges[0][1] if silent_ranges[0][0] == 0 else 0
        end_trim = silent_ranges[-1][0] if silent_ranges[-1][1] == len(audio) else len(audio)
    else:
        start_trim = 0
        end_trim = len(audio)
    return audio[max(start_trim - margin, 0):min(end_trim + margin, len(audio))]

def split_on_silence(audio, silence_threshold=-30, min_silence_len=500, margin=0):
	nonsilent_ranges = silence.detect_nonsilent(audio, silence_thresh=silence_threshold, min_silence_len=min_silence_len)
	nonsilent_ranges = [(max(x[0] - margin, 0), min(x[1] + margin, len(audio))) for x in nonsilent_ranges]
	return [audio[x[0]:x[1]] for x in nonsilent_ranges]

class AudioChunks():
	def __init__(self):
		self.items = [[0, AudioSegment.silent(1000)]]

	def add(self, start, audio):
		self.items.append([start, audio])

	@property
	def audio(self):
		result = AudioSegment.silent(0)
		for x in self.items:
			result = overlay([result, AudioSegment.silent(int(1000 * x[0])) + x[1]])
		return result
		#return overlay([AudioSegment.silent(int(1000 * x[0])) + x[1] for x in self.items])
