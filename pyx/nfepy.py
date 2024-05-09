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



def get_leaf_paths(element, current_path="", paths=None):
	if paths is None:
		paths = []
	path = f"{current_path}/{element.tag}" if current_path else element.tag
	if len(element) == 0:  # Leaf element
		paths.append(path)
	else:
		for child in element:
			get_leaf_paths(child, path, paths)
	return paths

def strip_namespace(tag):
	# Remove namespace if present
	return tag.split('}')[-1] if '}' in tag else tag

def xml2df(root):
	root = find(root, './/infNFe')
	leaf_paths = get_leaf_paths(root)
	#leaf_paths = [path.replace('{http://www.portalfiscal.inf.br/nfe}', '') for path in leaf_paths]
	leaf_paths = [path.split('infNFe/')[1] for path in leaf_paths]
	leaf_paths = sorted(list(set(leaf_paths)))

	#for path in leaf_paths: print(path)

	item_count = len(findall(root, './/det'))
	df = pd.DataFrame(data=[], columns=leaf_paths)#[path.split('/')[-1] for path in leaf_paths])
	for x in df.columns:
		leafs = root.findall(x) #Assuming it's ordered correctly
		#df[x] = pd.Series([leafs[0].text] * item_count if len(leafs) < item_count else [y.text for y in leafs])
		df[x] = pd.Series([y.text for y in leafs]).reindex(range(item_count))
		df = df.rename(columns={x: x.replace('{http://www.portalfiscal.inf.br/nfe}', '')}) #.split('/')[-1]}) DO IT AFTER CONCAT TO AVOID MISMATCH CAUSED BY DUPLICATED COLUMNS
	#print(df)
	return df

def xmls2df(ls):
	dfs = [xml2df(x).reset_index(drop=True) for x in ls]
	#columns = list(set.intersection(*[set(x.columns) for x in dfs]))
	#dfs = [x[columns].sort_index(axis=1) for x in dfs]
	df = pd.concat(dfs, axis=0, join='outer', ignore_index=True, sort=True)
