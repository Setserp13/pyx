from pypdf import PdfMerger

def merge_pdfs(files, output):
	merger = PdfMerger()
	for pdf in files:
		merger.append(pdf)
	merger.write(f'{output}.pdf')
	merger.close()