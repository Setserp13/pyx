from pyx.mat.rect import *

class Align:
	def align(obj, anchor, objPoint, anchorPoint): #both points are normalized with their respective rects
		return obj.setPosition(objPoint, anchor.denormalizePoint(anchorPoint))

	def above(obj, anchor): return Align.align(obj, anchor, Vector(.5, 0), Vector(.5, 1))
	def bellow(obj, anchor): return Align.align(obj, anchor, Vector(.5, 1), Vector(.5, 0))
	def onTheLeft(obj, anchor): return Align.align(obj, anchor, Vector(1, .5), Vector(0, .5))
	def onTheRight(obj, anchor): return Align.align(obj, anchor, Vector(0, .5), Vector(1, .5))

	def alignAxis(obj, anchor, axis, objPoint, anchorPoint):
		return obj.setAxisPosition(axis, objPoint, anchor.min[axis] + (anchor.size[axis] * anchorPoint))

	def leftToLeft(obj, anchor): return Align.alignAxis(obj, anchor, 0, 0, 0)
	def leftToRight(obj, anchor): return Align.alignAxis(obj, anchor, 0, 0, 1)
	def rightToLeft(obj, anchor): return Align.alignAxis(obj, anchor, 0, 1, 0)
	def rightToRight(obj, anchor): return Align.alignAxis(obj, anchor, 0, 1, 1)

	def bottomToBottom(obj, anchor): return Align.alignAxis(obj, anchor, 1, 0, 0)
	def bottomToTop(obj, anchor): return Align.alignAxis(obj, anchor, 1, 0, 1)
	def topToBottom(obj, anchor): return Align.alignAxis(obj, anchor, 1, 1, 0)
	def topToTop(obj, anchor): return Align.alignAxis(obj, anchor, 1, 1, 1)