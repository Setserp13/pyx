from copy import deepcopy
from io import BytesIO
import cairosvg
from lxml import etree
import pyx.svgx as svgx

#Convert from vector image to raster image

def svg_to_png(tree, output_png, rect=None, dpi=10):
	root = tree.getroot()
	if not rect is None:
		if rect.size[0] < 1 or rect.size[1] < 1: return #'CUZ NO SIZE CAN BE LESSER THAN 1
		root.attrib['viewBox'] = f"{rect.min[0]} {rect.min[1]} {rect.size[0]} {rect.size[1]}"	# Modify the SVG content to include a viewBox attribute
	svg_buffer = BytesIO()		# Create an in-memory file-like object
	tree.write(svg_buffer, encoding='utf-8', xml_declaration=True)
	svg_buffer.seek(0)	# Use the in-memory SVG content directly, no need for a temporary file
	cairosvg.svg2png(file_obj=svg_buffer, write_to=output_png, output_width=rect.size[0], output_height=rect.size[1], dpi=dpi)

def element_to_png(element, output_png, dpi=10):	#96):
	bbox = svgx.bbox(element)
	if not bbox:
		return

	new_svg = svgx.svg(100, 100)	# ========== BUILD A NEW SVG ==========
	new_svg.append(element)

	new_svg.attrib["viewBox"] = f"{bbox.min[0]} {bbox.min[1]} {bbox.size[0]} {bbox.size[1]}"
	new_svg.attrib["width"] = str(bbox.size[0])
	new_svg.attrib["height"] = str(bbox.size[1])

	# CONVERT
	svg_buffer = BytesIO()
	etree.ElementTree(new_svg).write(svg_buffer, encoding='utf-8', xml_declaration=True)
	svg_buffer.seek(0)

	cairosvg.svg2png(
		file_obj=svg_buffer,
		write_to=output_png,
		output_width=bbox.size[0],
		output_height=bbox.size[1],
		dpi=dpi,
	)
