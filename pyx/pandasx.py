from pandas import Series
from multipledispatch import dispatch
import pandas as pd
import pyx.array_utility as au
import os
import math
import numpy as np
import re
import pyx.lxmlx as lxmlx

def xml2df(root):
	lvs = lxmlx.leaves(root)
	series = {}
	for x in lvs:
		k = x['path'].replace('/', '_')
		if not k in series:
			series[k] = []
		series[k].append(x['element'])
	item_count = max([len(series[k]) for k in series])
	df = pd.DataFrame(data=[], columns=[k for k in series])
	for k in series:
		#print(k)
		df[k] = pd.Series([x.text for x in series[k]]).reindex(range(item_count))
	#print(df)
	return df


def isnull(value): return str(value) in ['None', 'NaN', 'NaT', 'none', 'nan', 'nat', ''] or pd.isnull(value)

def to_workbook(path, **worksheets):
	with pd.ExcelWriter(path) as writer:
		for k in worksheets:
			worksheets[k].to_excel(writer, sheet_name=re.sub(r'[\[\]:\*\?\\/]', '', k)[:31], index=False) #Remove invalid Excel character '[]:*?/\' in sheetname

def fetchall(df, **where): #where is a dict of (column, value)
	mask = df[where.keys()].eq([where[k] for k in where.keys()]).all(axis=1)
	return df[mask]

def fetchone(df, **where):
	result = fetchall(df, **where)
	return None if result.empty else result.iloc[0]
		
"""def select(df, columns, values):
	mask = df[columns].eq([values[column] for column in columns]).all(axis=1)
	return df[mask]"""

def segment(df, columns):
	#return [select(df, columns, row) for index, row in df[columns].drop_duplicates().iterrows()]
	return [fetchall(df, **dict(zip(columns, row))) for index, row in df[columns].drop_duplicates().iterrows()]

def foreach(df, columns, action): #action(df slice, column values)
	for index, row in df[columns].drop_duplicates().iterrows():
		#print(row)
		#action(select(df, columns, row), row)
		action(fetchall(df, **dict(zip(columns, row))), row)

def rename(df, columns): return df.rename(columns={x: columns[i] for i, x in enumerate(df.columns)})

#def segmentby(df, by):

@dispatch(pd.Series)
def strall(obj):
	obj = obj.map(lambda x: str(x))

@dispatch(pd.DataFrame)
def strall(obj):
	for x in obj.columns:
		strall(obj[x])



def readall(read, path_list, **kwargs): return pd.concat([read(x, **kwargs) for x in path_list])

def read_excels(dir, **kwargs):
	return pd.concat([pd.read_excel(os.path.join(dir, x), **kwargs) for x in os.listdir(dir) if x.endswith('.xlsx')])
		

def read_csvs(dir, **kwargs):
	return pd.concat([pd.read_csv(os.path.join(dir, x), **kwargs) for x in os.listdir(dir) if x.endswith('.csv')])

def ffill(arr, isnull=lambda x: x is None): return [arr[i-1] if i > 0 and isnull(arr[i]) else arr[i] for i in range(len(arr))]

def ffills(s): return pd.Series(ffill(s.tolist(), isnull=pd.isnull))

def ffilldf(df):
	for col in df.columns:
		df[col] = ffills(df[col])

def fillnext(df, columns):
	for i in range(df.shape[0] - 1):
		for col in columns:
			if pd.isnull(df[col].iloc[i + 1]) or df[col].iloc[i + 1] == 0:
				df[col].iloc[i + 1] = df[col].iloc[i]



def resum(df, new_sum, col): #SET SUM
	sum = df[col].sum()
	df[col] = df[col].map(lambda x: math.floor((x / sum) * new_sum))
	df[col].iloc[0] += new_sum - df[col].sum()
	#print(f'SUM IS {df[col].sum()}, {sum}')
	return df

def resume(df, by, agg):
	for x in by:
		df[x] = df[x].map(lambda x: str(x))
	lst = []

	cur = df
	
	for i in range(len(by)-1,-1,-1):
		cur = cur.groupby(by[:i+1]).agg(agg).reset_index()
		lst.append(cur)

	cur = cur.agg(agg)
	cur = pd.DataFrame({k:[v] for k,v in cur.items()})
	lst.append(cur)

	result = pd.concat(lst).sort_values(by)
	#result = result[by + list(agg.keys())]
	return result.groupby(by, dropna=False).agg(agg)






def insert(df, index, other): return pd.concat([df.iloc[:index], other, df.iloc[index:]]).reset_index(drop=True)

def insert_rows(df, index, *rows): return insert(df, index, pd.DataFrame(rows))

def slice(total, *qtys): return list(qtys) + [total - sum(qtys)]

def split_row(row, column, rels, *amount): #rels are columns dependent from column
	rels = [] if rels == None else rels
	weights = np.array(amount) / row[column]
	data = { column: slice(row[column], *amount) }
	rows = [row.copy() for x in data[column]]
	for x in rels:
		data[x] = slice(row[x], *np.floor(weights * row[x]))
	for k in data:
		for i, x in enumerate(data[k]):
			rows[i][k] = x
	return rows

def split_row_inplace(df, index, column, rels, *amount): #rels are columns dependent from column
	return insert_rows(df, index, *split_row(df.iloc[index], column, rels, *amount))

#def append(df, row): return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def extend(df, rows, columns=None): return pd.concat([df, pd.DataFrame(rows, columns=df.columns if columns is None else columns)], ignore_index=True)

def append(df, row, columns=None): return extend(df, [row], columns)

#CUIDADO PARA NÃO REPETIR O NOME DAS VARIÁVEIS QUE ITERAM EM LOOPS ANINHADOS
def pick(inventory, amount, column="OLDEGGS", rels=None):
	result = [ pd.DataFrame(columns=inventory.columns), pd.DataFrame(columns=inventory.columns) ]
	for i, row in inventory.iterrows():
		if amount > 0:
			if amount - row[column] < 0:
				for i, x in enumerate(split_row(row, column, rels, amount)):
					result[i] = append(result[i], x)
					#result[i] = result[i].append(x, ignore_index=True)
			else:
				result[0] = append(result[0], row.copy())
				#result[0] = result[0].append(row.copy(), ignore_index=True)
			amount -= row[column]
		else:
			result[1] = append(result[1], row.copy())
			#result[1] = result[1].append(row.copy(), ignore_index=True)
	return result
	


def split(df, vols, column, rels=None):
	if not isinstance(vols, list):
		vols = [math.floor(df[column].sum() / vols)] * (vols-1)
	result = [df]
	for i, x in enumerate(vols):
		picked = pick(result[0], int(float(x)), 'EST_SALEABLE_QTY', ['EGGS'])
		result[0] = picked[1]
		result.insert(1, picked[0])
	result.reverse()
	return result



def join(objs, on): #where len(objs) > 0
	result = objs[0]
	for i, x in enumerate(objs[1:], 1):
		result = pd.merge(result, x, on=on, how='outer', suffixes=('', f'_{i}'))
	return result

from pyx.sqlx import table_names

def read_db(con, table=None):
	return { x: pd.read_sql(f'SELECT * FROM [{x}]', con) for x in (table_names(con) if table == None else table) }
