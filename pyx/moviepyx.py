import moviepy.editor as mp
import pyx.osx as osx

def merge_av(audio_path, video_path, output_path):
	output_path = osx.to_distinct(video_path) if output_path == None else output_path
	audio = mp.AudioFileClip(audio_path)
	video = mp.VideoFileClip(video_path)
	video.set_audio(audio).write_videofile(output_path, fps=video.fps)

def set_video_duration(video_path, duration, speed_based=False, output_path=None):
	output_path = osx.to_distinct(video_path) if output_path == None else output_path
	video = VideoFileClip(video_path)
	video.fx(vfx.speedx, duration if speed_based else video.duration / duration).write_videofile(output_path)
	#return video
	#return output_path

















