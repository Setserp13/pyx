from pyx.mat.mat import rangef
from pyx.mat.rect import Rect2
from pyx.mat.vector import Vector

class Grid2:
	def __init__(self, cellSize, offset=Vector(0,0), cellGap=Vector(0,0)):
		self.cellSize = cellSize
		self.offset = offset
		self.cellGap = cellGap

	def cellToPoint(self, value):
		return self.offset + Vector.scale(self.cellSize + self.cellGap, value)

	def pointToCell(self, value):
		return Vector.floordiv((value - self.offset), self.cellSize + self.cellGap)
	
	def cell(self, i, j):
		return Rect2(*self.cellToPoint(Vector(i,j)), *self.cellSize)

	def byCellCount(size, cellCount): return grid2ByCellSize(size, Vector.divide(size, cellCount))

	def byCellSize(size, cellSize):
		result = []
		for x in rangef(0, size[0], cellSize[0]):
			for y in rangef(0, size[1], cellSize[1]):
				result.append(Rect2(x, y, min(cellSize[0], size[0] - x), min(cellSize[1], size[1] - y)))
		return result
