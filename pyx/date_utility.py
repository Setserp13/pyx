import datetime

def date(dt, format='yyyy-mm-dd'):
	dt = str(dt)
	year = substr(dt, strcoord(format, 'yyyy'))
	month = substr(dt, strcoord(format, 'mm'))
	day = substr(dt, strcoord(format, 'dd'))
	#print([year, month, day])
	return datetime.date(int(year), int(month), int(day))

def substr(str, coord=[None, None]): #coord has min_idx inclusive and max_idx exclusive
	if coord[0] == None: coord[0] = 0
	if coord[1] == None: coord[1] = len(str)
	#print(coord)
	return str[coord[0]:coord[1]]

def strcoord(str, substr):
	for i in range(len(str)):
		if str[i:i+len(substr)] == substr:
			return [i, i+len(substr)]
	return None


def date_range(start, count): return [start + datetime.timedelta(days=i) for i in range(count)]

def weekdays(dt): return date_range(dt.day - dt.isocalendar()[2], 7)

def first_date(year, month): return datetime.datetime(year, month, 1)

def last_date(year, month):
	if month == 12:
		return datetime.datetime(year, month, 31)
	else:
		return datetime.datetime(year, month + 1, 1) - datetime.timedelta(days = 1)

def prev_month():
	today = datetime.datetime.now()
	if today.month == 1:
		return (today.year - 1, 12)
	return (today.year, today.month - 1)

def join(objs, on): #where len(objs) > 0
	result = objs[0]
	for i, x in enumerate(objs[1:], 1):
		result = pd.merge(result, x, on=on, how='outer', suffixes=('', f'_{i}'))
	return result

