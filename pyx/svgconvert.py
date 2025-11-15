from copy import deepcopy
from io import BytesIO
import cairosvg
from lxml import etree
import pyx.svgx as svgx

def export_group_to_png(element, output_png, dpi=10):	#96):
	bbox = svgx.bbox(element)
	if not bbox:
		return

	new_svg = svgx.svg(0, 0)	# ========== BUILD A NEW SVG ==========
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
