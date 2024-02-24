import pyx.array_utility as au
from pyx.array_utility import sv
import sqlite3



def on_insert_restrict(local_table, local_column, foreign_table, foreign_column):
	return f"""CREATE TRIGGER {local_table}_{local_column}_on_insert_restrict
	BEFORE INSERT ON {local_table}
	FOR EACH ROW
	BEGIN
    		DECLARE count INTEGER;
		SELECT COUNT(*) INTO count FROM {foreign_table} WHERE {foreign_column} = NEW.{local_column};
		IF count = 0 THEN
			SELECT RAISE(ABORT, 'Invalid {local_column}: No matching {foreign_column} in {foreign_table}');
	END IF;
	END;"""


def delete_at(conn, table, id):
	conn.cursor().execute(f"DELETE FROM {table} WHERE id='{id}'")
	conn.commit()



#PRAGMA database_list

def database(cursor):
	database_path = cursor.execute('PRAGMA database_list').fetchall()[0][2]
	#print(database_path)
	database_filename = au.get_last(database_path.split('\\'))
	return database_filename[:len(database_filename) - 3]






		



def drop_column(cursor, table, column):
	cursor.execute(f"ALTER TABLE {table} DROP COLUMN {column}")





def add_column(cursor, table, **column_def_args):
	query = f"ALTER TABLE {table} ADD COLUMN " + column_def(**column_def_args)
	print(query)
	cursor.execute(query)




def tables(cursor, database_name=''):
	return cursor.execute("SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%'").fetchall()
	#return [x[0] for x in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall() if x[0] not in ['sqlite_sequence']]

def table_names(cursor): return [x[0] for x in tables(cursor)]

def table_info(cursor, table):
	return cursor.execute(f"PRAGMA table_info({table})").fetchall() #list of (cid  name  type  notnull  dflt_value  pk)

def column_info(cursor, table, column):
	for x in table_info(cursor, table):
		if x[1] == column:
			return x
	return None


def literal(value):
	try:
		return {
			bool:str(int(value)),
			bytes:f"X'{value.hex()}'",
			float:str(value),
			int:str(value),
			NoneType:"NULL",
			str:"'" + value.replace("'", "''") + "'"
		}[type(value)]
	except:
		raise ValueError("Unsupported type: {}".format(type(value)))

#args means positional arguments, kwargs means keyword arguments

def instances(cls, *tuple_list): return [cls(*x) for x in tuple_list]

def kwinstances(cls, *dict_list): return [cls(**x) for x in dict_list]


class column:
	def __init__(self, name, type, notnull=False, dflt_value=None, pk=False, autoincrement=False, unique=False, references=None): #references = (foreign_reference,) or (foreign_reference, on_delete) or (foreign_reference, on_delete, on_update)
		self.name = name
		self.type = type
		self.notnull = notnull
		self.dflt_value = dflt_value
		self.pk = pk
		self.autoincrement = autoincrement
		self.unique = unique
		self.fk = None if references is None else fk(self.name, *references)

	@property
	def sql(self):
		return f"{self.name} {self.type}{' not null' if self.notnull else ''}{f' default {self.dflt_value}' if self.dflt_value is not None else ''}{' primary key' if self.pk else ''}{' autoincrement' if self.autoincrement else ''}{' unique' if self.unique else ''}{self.fk.references if self.fk is not None else ''}"

def id(name='id'): return column(name, 'integer', pk=True, autoincrement=True)

def text_id(name='id'): return column(name, 'text', True, pk=True, unique=True)

def unique_name(name='name'): return column(name, 'text', True, unique=True)

def timestamp(name='timestamp'): return column(name, 'date', dflt_value='CURRENT_TIMESTAMP') #datetime type would be better

class fk:
	actions = ['', 'RESTRICT', 'SET NULL', 'SET DEFAULT', 'CASCADE'] #'' = 'NO ACTION'

	def __init__(self, local_column, foreign_reference, on_delete=0, on_update=0):
		self.local_column = local_column
		self.foreign_table, self.foreign_column = foreign_reference.split('.')
		self.on_delete = on_delete
		self.on_update = on_update

	@property
	def references(self):
		return f" REFERENCES {self.foreign_table}({self.foreign_column}){f' {actions[self.on_delete]} ON DELETE' if self.on_delete > 0 else ''}{f' {actions[self.on_update]} ON UPDATE' if self.on_update > 0 else ''}"

	@property
	def sql(self):
		return f"FOREIGN KEY ({self.local_column}){self.references}"

	@property
	def foreign_reference(self):
		return f'{self.foreign_table}.{self.foreign_column}'



class table:
	def __init__(self, name, objs, if_not_exists=True):	#columns, fks, if_not_exists=True):
		self.name = name
		self.objs = objs
		#self.columns = columns
		#self.fks = fks
		self.if_not_exists = if_not_exists

	@property
	def sql(self):
		return f"CREATE TABLE {'IF NOT EXISTS' if self.if_not_exists else ''} {self.name} ({sv([x.sql for x in self.objs])})"	#self.columns + self.fks])})"



"""class Parent:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

class Child(Parent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)"""



def columns(cursor, table_name): au.map(list(schema(cursor, table_name)), lambda x, i: x[1])

#column_path = table_name.column_name
def inner_join(cursor, column_paths='*', tables=[], join_conditions=[], condition='TRUE'):
	if column_paths == '*':
		columns = au.map(schema(cursor, table_name), lambda x, i: x[1])
	else:
		columns = column_paths.split(',')
	query = 'SELECT ' + column_paths + ' FROM '
	for i, x in enumerate(tables):
		if i > 0:
			query += ' INNER JOIN '
		query += x
		if i > 0:
			query += ' ON ' + join_conditions[i - 1]
	query += ' WHERE ' + condition
	print(query)
	return cursor.execute(query).fetchall()

def select(cursor, table_name, column_names='*', condition='TRUE'): #return rows as a dict array
	if column_names == '*':
		cols = columns(cursor, table_name)
	else:
		cols = column_names.replace(' ', '').split(',')
	rows = cursor.execute('SELECT ' + column_names + ' FROM ' + table_name + ' WHERE ' + condition).fetchall()
	return [{cols[i]:y for i, y in enumerate(x)} for x in rows]



def select_all(cursor, table, columns="*", condition='TRUE'):
	#print('SELECT ' + columns + ' FROM ' + table + ' WHERE ' + condition)
	return cursor.execute('SELECT ' + columns + ' FROM ' + table + ' WHERE ' + condition).fetchall()





def select_column(cursor, table, column): #return all column values as array
	return au.map(select_all(cursor, table, column), lambda x, i: x[0])



def columns(cursor, table): return [x[1] for x in table_info(cursor, table)]


def clear(cursor, table):
	cursor.execute(f"delete from {table} where TRUE")

def insert(con, tb, obj):
	columns = sv(list(obj.keys()))
	values = sv([x.text for x in list(obj.values())], quote="\'")
	query = f'INSERT INTO {tb} ({columns}) VALUES ({values})'
	con.cursor().execute(query)
	con.commit()

def insert_into(cursor, tb, obj):
	columns = sv(list(obj.keys()))
	values = sv(list(obj.values()), quote="\'")
	query = f'INSERT INTO {tb} ({columns}) VALUES ({values})'
	cursor.execute(query)

print('AAA')
"""
print('BBB')
def to_df(cursor, table, filename):
	#print(columns(cursor, table))
	data = select_all(cursor, table)
	df = pd.DataFrame(data, columns=dbu.columns(cursor, table))
	df.to_excel(filename)

	

def insertdf(cursor, table, df):
	matrix = df2mx(df)
	for row in matrix:
		values = sv(row, ",", "\'")
		print(values)
		cursor.execute("insert into " + table + " values(" + values + ")")



def insert(cursor, table, columns='', values='default values'):
	if columns != '': columns = '(' + columns + ')'
	if values != 'default values': values = 'values(' + sv(values, ',', '\'') + ')'
	sqlstr = 'insert into ' + table + ' ' + columns + ' ' + values
	#print(sqlstr)
	cursor.execute(sqlstr)



from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.checkbox import CheckBox
from alpha.kivyx import *
from kivy.properties import NumericProperty
from kivy.clock import Clock


class AutoFitSpinner(Spinner):
	max_width = NumericProperty(0)

	def on_text(self, instance, value):
		print(value)
		#self.max_width = max(self.max_width, self.texture_size[0])
		Clock.schedule_once(self.update_width, 0)

	def update_width(self, dt): self.max_width = self.texture_size[0] #max(self.max_width, self.texture_size[0])


	def on_max_width(self, instance, value):
		print(value)
		self.width = value
	

def structure(cursor, table, size_hint=(None, None), size=(100, 30)):
	layout = BoxLayout()
	grid = GridLayout(cols=6)
	grid.add_widgets(labels(['Name', 'Type', 'Primary Key', 'Default', 'Null', 'Action'], size_hint=size_hint, size=size))
	for x in table_info(cursor, table):
		grid.add_widgets([
			TextInput(text=x[1], size_hint=size_hint, size=size),
			AutoFitSpinner(text=x[2], values=['NULL', 'INTEGER', 'REAL', 'TEXT', 'BLOB'], size_hint=size_hint, size=size),
			CheckBox(active=x[3], size_hint=size_hint, size=size),
			TextInput(text=str(x[4]), size_hint=size_hint, size=size),
			CheckBox(active=x[5], size_hint=size_hint, size=size),
			Button(text='Drop', size_hint=size_hint, size=size)
		])
	layout.add_widget(grid)
	#button = Button(text='Add', size_hint=size_hint, size=size)

	#button.bind(on_press=lambda x: new_screen(lambda: Label(text='New screen')))
	#layout.add_widget(button)
	return layout



#def dfcols(df): return list(df.columns)

def dfrow(df, index): return [df.iat[index, j] for j in range(df.shape[1])]

def df2mx(df, include_columns = False):
	result = [dfrow(df, i) for i in range(df.shape[0])]
	if include_columns: return [ dfcols(df) ] + result
	return result







from alpha.xl import *
#export/import

def database_to_xlsx(cursor, db=''):
	wb = EmptyWorkbook()
	for tb in tables(cursor): add_worksheet(wb, table_to_xlsx(cursor, tb))
	return wb

def export_database_as_xlsx(cursor, db='', dst_filename=None):
	if dst_filename == None: dst_filename = database(cursor) + '.xlsx'
	database_to_xlsx(cursor, db).save(filename = dst_filename)

def table_to_xlsx(cursor, tb):
	ws = Worksheet()
	ws.title = tb
	set_row(ws, 1, columns(cursor, tb))
	set_rng(ws, sel2mx(select_all(cursor, tb)), min_row=2, min_col=1)
	return ws

def export_table_as_xlsx(cursor, tb, dst_filename): table_to_xlsx(cursor, tb).parent.save(filename = dst_filename)
"""
