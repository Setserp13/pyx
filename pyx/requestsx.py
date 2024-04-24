import os
import pyx.osx as osx
import requests
from urllib.parse import urlparse

def download_image(url, filename=None, dir=None):
	try:
		response = requests.get(url)
		if response.status_code == 200:
			urlpath = urlparse(url).path
			basename = f'{osx.filename(urlpath) if filename == None else filename}{osx.ext(urlpath)}'
			file = osx.to_distinct(basename if dir == None else os.path.join(dir, basename))
			osx.writeb(file, response.content)
			print("Image downloaded successfully")
		else:
			print(f"Failed to download image. Status code: {response.status_code}")
	except Exception as e:
		print(f"An error occurred: {e}")
