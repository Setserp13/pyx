import functools
from pydub import AudioSegment

AudioSegment.merge = lambda ls: functools.reduce(lambda x, y: x+y, ls)#, AudioSegment.silent(0))
