import numpy as np
from PIL import Image
import math
import random
import pyx.numpyx as npx

def perlin_noise(width, height, scale=20, seed=None):
	"""
	Gera uma imagem de Perlin Noise usando numpy e PIL.

	width, height -> tamanho da imagem
	scale -> tamanho das células da grid
	seed -> semente opcional
	show -> se True, exibe a imagem
	"""

	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)

	# Número de pontos de grade
	grid_x = width // scale + 2
	grid_y = height // scale + 2

	# Gera gradientes aleatórios unitários para cada ponto da grade
	gradients = np.zeros((grid_y, grid_x, 2))
	for y in range(grid_y):
		for x in range(grid_x):
			angle = random.uniform(0, 2 * math.pi)
			gradients[y, x] = np.array([math.cos(angle), math.sin(angle)])

	# Função de fade (curva suave)
	def fade(t):
		return 6*t**5 - 15*t**4 + 10*t**3

	# Imagem resultante
	img = np.zeros((height, width), dtype=np.float32)

	for y in range(height):
		for x in range(width):

			# Posição dentro da célula
			gx = x // scale
			gy = y // scale

			# Coordenadas locais
			tx = (x % scale) / scale
			ty = (y % scale) / scale

			# Vetores para os 4 cantos
			v00 = np.array([tx, ty])
			v10 = np.array([tx - 1, ty])
			v01 = np.array([tx, ty - 1])
			v11 = np.array([tx - 1, ty - 1])

			# Produtos escalares com gradientes
			d00 = gradients[gy, gx].dot(v00)
			d10 = gradients[gy, gx+1].dot(v10)
			d01 = gradients[gy+1, gx].dot(v01)
			d11 = gradients[gy+1, gx+1].dot(v11)

			# Aplica fade
			fx = fade(tx)
			fy = fade(ty)

			# Interpolação nos eixos X e Y
			lx0 = npx.lerp(d00, d10, fx)
			lx1 = npx.lerp(d01, d11, fx)
			img[y, x] = npx.lerp(lx0, lx1, fy)

	# Normaliza para 0–255
	img = (img - img.min()) / (img.max() - img.min())
	img = (img * 255).astype(np.uint8)

	image = Image.fromarray(img, mode="L")

	return image
