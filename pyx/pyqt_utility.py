from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
#from alpha.array_utility import *
import alpha.array_utility as au

from alpha.matrix_utility import MatrixLike

def countRow(grid): return int(grid.count() / grid.columnCount())

def QGridLayoutMatrix(grid):
	return MatrixLike(grid, lambda x: [countRow(x), x.columnCount()], lambda x, i, j: x.itemAtPosition(i, j), lambda x, i, j, v: x.addItem(v, i, j))



def forRows(grid, func, min_row=1):
	for i in range(min_row, countRow(grid)):
		func(itemsAtRow(grid, i), i, grid)




def itemsInRange(grid, min_row=0, min_col=0, row_count=None, col_count=None):
	if row_count == None: row_count = countRow(grid)
	if col_count == None: col_count = grid.columnCount()
	result = []
	for i in range(row_count):
		for j in range(col_count):
			result.append(grid.itemAtPosition(min_row + i, min_col + j))
	return result

def itemsAtRow(grid, row): return itemsInRange(grid, min_row=row, row_count=1)

def itemsAtColumn(grid, col): return itemsInRange(grid, min_col=col, col_count=1)

def positionOf(grid, item):
	idx = grid.indexOf(item)
	if idx > -1: return grid.getItemPosition(idx)
	return None




def getLabels(grid): return au.map(itemsAtRow(grid, 0), lambda x: x.widget().text())

def getValuesAtRow(grid, row): return au.map(itemsAtRow(grid, row), lambda x: x.widget().text())


def labelsToNumbers(grid, labels):
	allLabels = getLabels(grid)
	return au.map(labels, lambda x: au.index_of(allLabels, x))

def labelsToItemsAtRow(grid, row, labels):
	return au.get_items(itemsAtRow(grid, row), labelsToNumbers(grid, labels))

def labelsToValuesAtRow(grid, row, labels):
	return au.map(labelsToItemsAtRow(grid, row, labels), lambda x: x.widget().text())


def addLabels(grid, labels):
	for i in range(len(labels)):
		label = QLabel(labels[i])
		grid.addWidget(label, 0, i)#[, alignment=0])
		#print(positionOf(grid, label))

def addWidgetRow(grid, widgets, min_row=None, min_col=0):
	if min_row == None: min_row = countRow(grid)
	#print(min_row)
	for i in range(len(widgets)): grid.addWidget(widgets[i], min_row, min_col + i)#[, alignment=0])

def removeRowAt(grid, idx):
	for i in range(grid.columnCount() - 1, -1, -1):
		#print(i)
		x = grid.itemAtPosition(idx, i)
		#print(x)
		#del x
		#grid.takeAt(grid.indexOf(x))
		grid.removeItem(x)
		#x.hide()
		x.widget().deleteLater()
	for i in range(idx + 1, grid.rowCount()):#countRow(grid)):
		for j in range(0, grid.columnCount()):
			setItemPosition(grid, grid.itemAtPosition(i, j), i - 1, j)

def setItemPosition(grid, item, *pos):
	if item == None: return
	grid.removeItem(item)
	grid.addItem(item, pos[0], pos[1])




def clear(grid):
	for i in range(countRow(grid) - 2):
		removeRowAt(grid, 1) #it keeps the last row

def removeRow(grid, row):
	for x in row:
		#print(x)
		grid.removeWidget(x)
		#x.hide()
		x.deleteLater()

from alpha.database_utility import export_database_as_xlsx
def saveGrid(grid, saveGridRow, conn):
	forRows(grid, lambda x, i, grid: saveGridRow(x, i, grid))
	conn.commit()
	clear(grid)
	export_database_as_xlsx(conn.cursor())
	print('Done!')

def dynamicGrid(labels, row_maker, row_saver, c): #each row must have the same size as labels
	vb = QVBoxLayout()
	grid = QGridLayout()
	addLabels(grid, labels)
	row = addDynamicRow(grid, row_maker)
	launchbtn = QPushButton('LANÇAR')
	#launchbtn.clicked.connect(lambda: print('click!'))
	launchbtn.clicked.connect(lambda: saveGrid(grid, row_saver, c))
	vb.addLayout(grid)
	vb.addWidget(launchbtn)
	return vb


def addDynamicRow(grid, func, min_row=None, min_col=0): #func returns a new row
	row = func()
	addWidgetRow(grid, row, min_row, min_col)
	for w in row:
		#w.editingFinished.connect(lambda:print("Editing finished"))
                w.editingFinished.connect(lambda: tryExpandFrom(grid, row, func))
		#w.textChanged.connect(lambda: tryExpandFrom(grid, row, func))

def tryExpandFrom(grid, row, func):
	row_idx = positionOf(grid, row[0])[0]
	#print('last row ' + str(countRow(grid) - 1) + ', current row ' + str(row_idx))
	if row_idx == countRow(grid) - 1: #if is the last row
		if any(x.text() != '' for x in row):
			addDynamicRow(grid, func)
	if all(x.text() == '' or x.text() == '0' for x in row) and countRow(grid) > 1:
		removeRowAt(grid, row_idx)
		#removeRow(grid, row)
	#print(countRow(grid))



def addWidgetMatrix(grid, widgets, min_row=None, min_col=0):
	if min_row == None: min_row = countRow(grid)
	for i in range(len(widgets)):
		for j in range(len(widgets[i])):
			grid.addWidget(widgets[i][j], min_row + i, min_col + j)#[, alignment=0])


"""
Dynamic grid behaviour
-Whenever a non-default value is set in any cell in the last row of the grid, a new default row is pushed on the grid
-Whenever a non-last and non-focused row in the grid becomes a default row, it is removed from the grid














"""