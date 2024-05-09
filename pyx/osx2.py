import pyx.osx as osx
from pydub import AudioSegment
import cv2

def open_audios(path_list, **kwargs): return osx.open_all(path_list, open_audio, **kwargs)

def open_videos(path_list, **kwargs): return osx.open_all(path_list, cv2.VideoCapture, **kwargs)

def open_audio(path):
	return AudioSegment.from_file(path, format=ext(path)[1:])

def export_audio(audio, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	audio.export(path, format=osx.ext(path)[1:])

"""def open_audios_from(dir):
	return open_audios([os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.mp3') or x.endswith('.wav')])

def open_videos_from(dir):
	return open_videos([os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.mp4')])"""