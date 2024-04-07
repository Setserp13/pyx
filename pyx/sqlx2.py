from pyx.array_utility import sv
import sqlite3
import pandas as pd



def values(df): return sv(["(" + sv(list(x), quote="'") + ")" for i, x in df.iterrows()])

def insert_into(conn, table, df):
	sql = f"INSERT INTO {table} VALUES {values(df)}"
	#print(sql)
	conn.cursor().execute(sql)
	conn.commit()


def insert_into_columns(conn, table, df):
	columns = sv(list(df.columns))
	sql = f"INSERT INTO {table} ({columns}) VALUES "
	for i, x in df.iterrows():
		try:
			conn.cursor().execute(sql + "(" + sv(list(x), quote="'") + ")")	
		except: pass
	conn.commit()


def load_data(conn, filepath):
	xls = pd.ExcelFile(filepath)
	for sheet_name in xls.sheet_names:
		insert_into_columns(conn, sheet_name, xls.parse(sheet_name))

