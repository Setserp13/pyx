from pydub import AudioSegment

AudioSegment.merge = lambda ls: reduce(lambda x, y: x+y, ls)#, AudioSegment.silent(0))