import re
import xml.etree.ElementTree as ET

def search(root, match):
	#print(re.sub('/(?!/)', '/ns:', match))
	obj = root.find(re.sub('/(?!/)', '/ns:', match), { 'ns' : 'http://www.portalfiscal.inf.br/nfe' })
	return '' if obj == None else obj.text

def findall(root, match):
	#print(re.sub('/(?!/)', '/ns:', match))
	return root.findall(re.sub('/(?!/)', '/ns:', match), { 'ns' : 'http://www.portalfiscal.inf.br/nfe' })

prodTags = ['cProd', 'cEAN', 'xProd', 'NCM', 'CEST', 'indEscala', 'CFOP', 'uCom', 'qCom', 'vUnCom', 'vProd', 'cEANTrib', 'uTrib', 'qTrib', 'vUnTrib', 'indTot']

def prods(root, tags=prodTags):
	result = []
	for i, x in enumerate(zip(*[findall(root, f'.//{y}') for y in tags])):
		result.append({tags[j]: float(y.text) if '.' in y.text else y.text for j, y in enumerate(x)})
	return result