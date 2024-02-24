import pyautogui
import time
import pandas as pd
import datetime
from inspect import isfunction


def click(x, y, times=1):
	pyautogui.moveTo(x, y)
	for i in range(times): pyautogui.click()

def locatePivotOnScreen(img, pivot=(0.5, 0.5), confidence=1):
	rect = pyautogui.locateOnScreen(img, confidence)
	#print(img)
	if rect != None:
		return (rect[0] + rect[2] * pivot[0], rect[1] + rect[3] * pivot[1])
	

	#left, top, width, height = pyautogui.locateOnScreen(img, confidence)
	#return (left + width * pivot[0], top + height * pivot[1])


def clickOn(img, pivot=(0.5, 0.5), confidence=1, times=1):
	pos = None
	while pos == None:
		pos = locatePivotOnScreen(img, pivot, confidence)
	click(*pos, times)




def scrollTo(dir, tar): #SCROLLS TO TARGET IMAGE APPEARS
	pos = None
	while pos == None:
		pos = locatePivotOnScreen(tar[0], tar[1], tar[2] if len(tar) > 2 else 1)
		pos2 = None
		while pos2 == None:
			pos2 = locatePivotOnScreen(dir[0], dir[1])
		click(*pos2, 1)
		pyautogui.move(-100,0)
	return pos



def click_some(imgs, times=1):
	pos = None
	while pos == None:
		for x in imgs:
			pos = locatePivotOnScreen(x[0], x[1], x[2] if len(x) > 2 else 1)
			if pos != None:
				break
	click(*pos, times)		


#pyautogui.size() #screen size
#print(pyautogui.position())

def run_sub(sub):
	for x in sub:
		if isinstance(x, list):
			clickOn(x[0], x[1], x[3] if len(x) > 3 else 1, x[2] if len(x) > 2 else 1)

			"""pos = None
			while pos == None:
				if len(x) > 3:
					print(x[3])
					pos = locatePivotOnScreen(x[0], x[1], x[3])
				else:
					pos = locatePivotOnScreen(x[0], x[1])
			times = 1
			if len(x) > 2:
				times = x[2]
			click(*pos, times)"""

		elif isinstance(x, str):
			pyautogui.typewrite(x)
		elif isinstance(x, int):
			time.sleep(x)
		elif isfunction(x):
			x()

