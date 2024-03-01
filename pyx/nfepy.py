import re
import xml.etree.ElementTree as ET

def search(root, match):
	return getattr(find(root, match), 'text', None)

def find(root, match):
	return root.find(re.sub('/(?!/)', '/ns:', match), { 'ns' : 'http://www.portalfiscal.inf.br/nfe' })
	
def findall(root, match):
	#print(re.sub('/(?!/)', '/ns:', match))
	return root.findall(re.sub('/(?!/)', '/ns:', match), { 'ns' : 'http://www.portalfiscal.inf.br/nfe' })

prodTags = ['cProd', 'cEAN', 'xProd', 'NCM', 'CEST', 'indEscala', 'CFOP', 'uCom', 'qCom', 'vUnCom', 'vProd', 'cEANTrib', 'uTrib', 'qTrib', 'vUnTrib', 'indTot']

def prods(root, tags=prodTags):
	result = []
	for i, x in enumerate(zip(*[findall(root, f'.//{y}') for y in tags])):
		result.append({tags[j]: float(y.text) if '.' in y.text else y.text for j, y in enumerate(x)})
	return result
