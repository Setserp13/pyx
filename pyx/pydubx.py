import functools
from pydub import AudioSegment

AudioSegment.merge = lambda ls: functools.reduce(lambda x, y: x+y, ls)#, AudioSegment.silent(0))

def overlay(ls): return functools.reduce(lambda a, b: a.overlay(b) if len(a) > len(b) else b.overlay(a), ls)

def open_all(ls): return [AudioSegment.from_file(x) for x in ls]
