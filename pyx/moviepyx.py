import moviepy.editor as mp

def merge_av(audio_path, video_path, output_path):
	audio = mp.AudioFileClip(audio_path)
	video = mp.VideoFileClip(video_path)
	video.set_audio(audio).write_videofile(output_path, fps=video.fps)

















