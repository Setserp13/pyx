import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

class SQLiteDB:
	def __init__(self, database: str):
		self.database = Path(database)
		self.connection = sqlite3.connect(self.database)
		self.connection.row_factory = sqlite3.Row
		self.cursor = self.connection.cursor()

	# ----------------------------------------------------
	# Connection
	# ----------------------------------------------------

	def close(self): self.connection.close()

	def commit(self): self.connection.commit()

	def rollback(self): self.connection.rollback()

	# ----------------------------------------------------
	# Generic Queries
	# ----------------------------------------------------

	def execute(self, query: str, params: Iterable = ()):
		self.cursor.execute(query, params)
		self.commit()
		return self.cursor

	def executemany(self, query: str, data):
		self.cursor.executemany(query, data)
		self.commit()

	def fetchone(self, query: str, params: Iterable = ()) -> Optional[dict]:
		cur = self.cursor.execute(query, params)
		row = cur.fetchone()
		return dict(row) if row else None

	def fetchall(self, query: str, params: Iterable = ()) -> list[dict]:
		cur = self.cursor.execute(query, params)
		return [dict(r) for r in cur.fetchall()]

	# ----------------------------------------------------
	# CRUD Helpers
	# ----------------------------------------------------

	def insert(self, table: str, values: dict):
		columns = ", ".join(values.keys())
		placeholders = ", ".join("?" for _ in values)

		sql = f"""
		INSERT INTO {table} ({columns})
		VALUES ({placeholders})
		"""

		self.execute(sql, tuple(values.values()))
		return self.cursor.lastrowid

	def update(self, table: str, values: dict, where: str, where_params=()):
		set_clause = ", ".join(f"{k}=?" for k in values.keys())

		sql = f"""
		UPDATE {table}
		SET {set_clause}
		WHERE {where}
		"""

		params = tuple(values.values()) + tuple(where_params)
		self.execute(sql, params)

	def upsert(self, table: str, values: dict, conflict):
		if isinstance(conflict, str):
			conflict = [conflict]

		columns = ", ".join(values.keys())
		placeholders = ", ".join("?" for _ in values)

		update_clause = ", ".join(
			f"{k}=excluded.{k}"
			for k in values.keys()
			if k not in conflict
		)

		conflict_clause = ", ".join(conflict)

		sql = f"""
		INSERT INTO {table} ({columns})
		VALUES ({placeholders})
		ON CONFLICT({conflict_clause}) DO UPDATE SET
			{update_clause}
		"""

		self.execute(sql, tuple(values.values()))
		return self.cursor.lastrowid
	
	def delete(self, table: str, where: str, params=()):
		sql = f"DELETE FROM {table} WHERE {where}"
		self.execute(sql, params)

	def select(
		self,
		table: str,
		columns="*",
		where=None,
		params=(),
		order_by=None,
		limit=None,
	):
		sql = f"SELECT {columns} FROM {table}"

		if where: sql += f" WHERE {where}"

		if order_by: sql += f" ORDER BY {order_by}"

		if limit: sql += f" LIMIT {limit}"

		return self.fetchall(sql, params)

	def search(
		self,
		table: str,
		search_fields: list[str],
		search: str = "",
		where: dict = None,
		order_by: str = None,
		page: int = 1,
		limit: int = 50,
	):
		sql = f"SELECT * FROM {table}"
		params = []
	
		conditions = []
	
		if search:
			parts = []
	
			for field in search_fields:
				parts.append(f"{field} LIKE ?")
				params.append(f"%{search}%")
	
			conditions.append("(" + " OR ".join(parts) + ")")
	
		if where:
			for field, value in where.items():
				conditions.append(f"{field} = ?")
				params.append(value)
	
		if conditions:
			sql += " WHERE " + " AND ".join(conditions)
	
		if order_by:
			sql += f" ORDER BY {order_by}"
	
		sql += " LIMIT ? OFFSET ?"
	
		params.extend([
			limit,
			(page - 1) * limit
		])
	
		items = self.fetchall(sql, params)
	
		# count
		count_sql = f"SELECT COUNT(*) AS total FROM {table}"
		count_params = []
	
		if conditions:
			count_sql += " WHERE " + " AND ".join(conditions)
	
			# rebuild params without limit/offset
			count_params = params[:-2]
	
		total = self.fetchone(
			count_sql,
			count_params
		)["total"]
	
		return {
			"items": items,
			"total": total,
			"page": page,
			"pages": (total + limit - 1) // limit
		}
	# ----------------------------------------------------
	# Table Utilities
	# ----------------------------------------------------

	def create_table(self, table: str, **columns):
		cols = ", ".join(
			f"{name} {definition}"
			for name, definition in columns.items()
		)
		sql = f"""
		CREATE TABLE IF NOT EXISTS {table} (
			{cols}
		)
		"""
		self.execute(sql)

	def drop_table(self, table: str): self.execute(f"DROP TABLE IF EXISTS {table}")

	def table_exists(self, table: str) -> bool:
		row = self.fetchone(
			"""
			SELECT name
			FROM sqlite_master
			WHERE type='table'
			AND name=?
			""",
			(table,),
		)
		return row is not None

	def list_tables(self):
		rows = self.fetchall("""
			SELECT name
			FROM sqlite_master
			WHERE type='table'
			ORDER BY name
		""")
		return [r["name"] for r in rows]

	def get_columns(self, table: str): return self.fetchall(f"PRAGMA table_info({table})")

	# ----------------------------------------------------
	# Schema Editing
	# ----------------------------------------------------

	def add_column(self, table: str, column: str, datatype: str):
		sql = f"""
		ALTER TABLE {table}
		ADD COLUMN {column} {datatype}
		"""
		self.execute(sql)

	def rename_table(self, old_name: str, new_name: str):
		self.execute(
			f"ALTER TABLE {old_name} RENAME TO {new_name}"
		)

	def rename_column(self, table: str, old: str, new: str):
		self.execute(
			f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}"
		)

	# ----------------------------------------------------
	# Metadata
	# ----------------------------------------------------

	def count(self, table: str):
		row = self.fetchone(f"SELECT COUNT(*) AS total FROM {table}")
		return row["total"]

	def vacuum(self): self.execute("VACUUM")

	def integrity_check(self):
		row = self.fetchone("PRAGMA integrity_check")
		return row["integrity_check"]

	# ----------------------------------------------------
	# Context Manager
	# ----------------------------------------------------

	def __enter__(self): return self

	def __exit__(self, exc_type, exc, tb):
		if exc:
			self.rollback()
		else:
			self.commit()

		self.close()
