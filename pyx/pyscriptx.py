import io
from js import document, File, Uint8Array, URL
import pandas as pd
from tempfile import NamedTemporaryFile
import pyx.osx as osx

async def FileList2BytesIO(obj): return [await to_bytes(obj.item(i)) for i in range(obj.length)]

async def to_bytes(file):
	array_buf = Uint8Array.new(await file.arrayBuffer())
	bytes = bytearray(array_buf)
	return io.BytesIO(bytes)
	
def to_file(bytes, filename):
	bytes.seek(0)
	js_array = Uint8Array.new(bytes.getbuffer())
	return File.new([js_array], filename, {type: osx.ext(filename)})	#{type: ".xlsx"})
	
def download(file, filename):
	url = URL.createObjectURL(file)
	hidden_link = document.createElement("a")
	hidden_link.setAttribute("download", filename)
	hidden_link.setAttribute("href", url)
	hidden_link.click()

def to_excel(df, filename): #DOWNLOAD AS EXCEL
	buffer = io.BytesIO()
	with pd.ExcelWriter(buffer) as writer:
		df.to_excel(writer)
	print('AAA')
	file = to_file(buffer, filename)
	#buffer.seek(0)
	#js_array = Uint8Array.new(buffer.getbuffer())
	#file = File.new([js_array], filename, {type: ".xlsx"})
	download(file, filename)	
		
def to_txt(text, filename): #DOWNLOAD AS TXT
	encoded_data = text.encode('utf-8')
	my_stream = io.BytesIO(encoded_data)
	file = to_file(my_stream, filename)
	#print('TO TEXT')
	#js_array = Uint8Array.new(len(encoded_data))
	#js_array.assign(my_stream.getbuffer())
	#file = File.new([js_array], filename, {type: "text/plain"})
	download(file, filename)

def save_excel(wb, filename): #DOWNLOAD AS EXCEL
	with NamedTemporaryFile() as tmp:
		wb.save(tmp.name)
		tmp.seek(0)
		bytes = tmp.read()
		js_array = Uint8Array.new(bytes)
		file = File.new([js_array], filename, {type: ".xlsx"})
		download(file, filename)
