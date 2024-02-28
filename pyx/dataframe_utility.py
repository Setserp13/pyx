from pandas import Series
from multipledispatch import dispatch
import pandas as pd
import pyx.array_utility as au
import os
import math

def select(df, columns, values):
	match = True
	for x in columns:
		match = match & (df[x] == values[x])
	#print(df[match])
	return df[match]

def foreach(df, columns, action): #action(df slice, column values)
	for index, row in df[columns].drop_duplicates().iterrows():
		#print(row)
		action(select(df, columns, row), row)

#def segmentby(df, by):








@dispatch(pd.Series)
def strall(obj):
	obj = obj.map(lambda x: str(x))

@dispatch(pd.DataFrame)
def strall(obj):
	for x in obj.columns:
		strall(obj[x])

def row_of(df, dict): #dict is like { column1: value1, ..., column2: value2 }
	for i in range(df.shape[0]):
		match = True
		for key, value in dict.items():
			if df[key][i] != value:
				match = False
				break
		if match: return i
	return -1

def find_rows(df, match, columns = None):
	result = []
	for index, row in df.iterrows():
		if match(row):
			if columns == None:            
				result.append(row)
			else:
				result.append(list(au.map(columns, lambda x, i: row[x])))
	return result


def readall(read, path_list, **kwargs): pd.concat([read(x, **kwargs) for x path_list])

def read_excels(dir, **kwargs):
	return pd.concat([pd.read_excel(os.path.join(dir, x), **kwargs) for x in os.listdir(dir) if x.endswith('.xlsx')])
		

def read_csvs(dir, **kwargs):
	return pd.concat([pd.read_csv(os.path.join(dir, x), **kwargs) for x in os.listdir(dir) if x.endswith('.csv')])


def fillnext(df, columns):
	for i in range(df.shape[0] - 1):
		for col in columns:
			if pd.isnull(df[col].iloc[i + 1]) or df[col].iloc[i + 1] == 0:
				df[col].iloc[i + 1] = df[col].iloc[i]

"""
def weights(df, key_columns, value_column):
	result = result.groupby(by=key_columns).sum(numeric_only = True).reset_index()
	total = result[value_column].sum()
	result[value_column] = result[value_column].map(lambda x: x / total)
	return result

def absolute(df, key_columns, value_column, total):
	result = df
	result[value_column] = result[value_column].map(lambda x: floor(x * total))
	result[value_column][-1] += total - result[value_column].sum()
	return result
"""

def resum(df, new_sum, col): #SET SUM
	sum = df[col].sum()
	df[col] = df[col].map(lambda x: math.floor((x / sum) * new_sum))
	df[col].iloc[0] += new_sum - df[col].sum()
	#print(f'SUM IS {df[col].sum()}, {sum}')
	return df








#CUIDADO PARA NÃO REPETIR O NOME DAS VARIÁVEIS QUE ITERAM EM LOOPS ANINHADOS

def pick(inventory, amount, column="OLDEGGS", rels=None):
	result = [ pd.DataFrame(columns=inventory.columns),	pd.DataFrame(columns=inventory.columns) ]
	for i in range(inventory.shape[0]):
		row = inventory.iloc[[i]]
		if amount > 0:
			qty = inventory.columns.get_loc(column)
			row.iloc[0, qty] = min(row.iloc[0, qty], amount)
			total = inventory.iloc[i, qty]
			amount = amount - row.iloc[0, qty]
			if amount <= 0:
				result.append(pd.DataFrame(columns=inventory.columns))
				row2 = inventory.iloc[[i]]
				row2.iloc[0, qty] = inventory.iloc[i, qty] - row.iloc[0, qty]
				
				if rels != None:
					for j in range(len(rels)):
						rel = inventory.columns.get_loc(rels[j])
						#print([inventory.iloc[i, rel], row.iloc[0, rel] ])
						if total != 0:
							row.iloc[0, rel] = math.floor((row.iloc[0, qty] / total) * row.iloc[0, rel])
							row2.iloc[0, rel] = inventory.iloc[i, rel] - row.iloc[0, rel]

				result[1] = pd.concat([result[1], row2])
			result[0] = pd.concat([result[0], row])
		else:
			result[1] = pd.concat([result[1], row])


	return result




def split(df, n, column, rels):
	result = []
	term = math.floor(df[column].sum() / n)
	#print(term)
	for i in range(n - 1):
		picked = pick(df, term, column, rels)
		result.append(picked[0])
		df = picked[1]
		#print(picked[1][column].sum())
	result.append(df)
	return result




