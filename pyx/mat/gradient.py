from pyx.mat.mat import lerp

def gradient(arr, t): #arr is a list of (color, pos)
	#print(arr)
	for i in range(len(arr) - 1):
		if t >= arr[i][1] and t <= arr[i+1][1]:
			dist = arr[i+1][1] - arr[i][1]
			if dist == 0:
				return arr[i][0]
			else:
				return lerp(arr[i][0], arr[i + 1][0], (t - arr[i][1]) / dist)
	if len(arr) == 0:
		return (0.0,0.0,0.0,1.0)
	elif len(arr) == 1 or t < arr[0][1]:
		return arr[0][0]
	else:
		return arr[-1][0]


"""def gradient(arr, t): #arr is a list of color
	step = 1.0 / (len(arr) - 1)
	for i in range(len(arr) - 1):
		if t >= step * i and t <= step * (i+1):
			return lerp(arr[i], arr[i + 1], (t - step * i) * (len(arr) - 1))"""