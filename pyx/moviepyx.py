import moviepy.editor as mp
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, vfx
import pyx.osx as osx
import os
from pydub import AudioSegment, silence

def merge_av(audio_path, video_path, output_path):
	output_path = osx.to_distinct(video_path) if output_path == None else output_path
	audio = mp.AudioFileClip(audio_path)
	video = mp.VideoFileClip(video_path)
	video.set_audio(audio).write_videofile(output_path, fps=video.fps)

def set_duration(video, duration, speed_based=False): return video.fx(vfx.speedx, duration if speed_based else video.duration / duration)

def set_video_duration(video_path, duration, speed_based=False, output_path=None):
	output_path = osx.to_distinct(video_path) if output_path == None else output_path
	video = VideoFileClip(video_path)
	set_duration(video, duration, speed_based).write_videofile(output_path)
	#return video
	#return output_path

def remove_silence(video, silence_threshold=-60, min_silence_len=1500):
	audio = video.audio
	temp_path = osx.to_distinct('TEMP.wav')
	audio.write_audiofile(temp_path)#, codec='pcm_s16le')
	audio_segment = AudioSegment.from_wav(temp_path)
	nonsilent_ranges = silence.detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_threshold)
	os.remove(temp_path)
	clips = []
	for start, end in nonsilent_ranges:
		clip = video.subclip(start / 1000.0, end / 1000.0)
		clips.append(clip)
	return concatenate_videoclips(clips)

def repeat_clip(clip, duration):
	clip_duration = clip.duration
	if clip_duration > duration:
		return clip.subclip(0, duration)
	elif clip_duration == duration:
		return clip
	else:
		repeats = int(duration // clip_duration)
		remainder = duration % clip_duration
		clips = [clip] * repeats
		if remainder > 0:
			clips.append(clip.subclip(0, remainder))
		return concatenate_videoclips(clips, method="compose")

def overlay_videoclips(clips, duration=None):
	duration = max([x.duration for x in clips]) if duration is None else duration
	return CompositeVideoClip([x.set_start(0) for x in clips]).subclip(0, duration)

def add_audio_to_video(video, audio): return video.set_audio(audio.subclip(0, video.duration))














