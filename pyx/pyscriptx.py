import io
from js import File, Uint8Array, URL
import pandas as pd

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
	buffer.seek(0)
	js_array = Uint8Array.new(buffer.getbuffer())
	file = File.new([js_array], filename, {type: ".xlsx"})
	download(file, filename)	
		
def to_txt(text, filename): #DOWNLOAD AS TXT
	encoded_data = text.encode('utf-8')
	my_stream = io.BytesIO(encoded_data)
	js_array = Uint8Array.new(len(encoded_data))
	js_array.assign(my_stream.getbuffer())
	file = File.new([js_array], filename, {type: "text/plain"})
	download(file, filename)