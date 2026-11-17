"""SQL execution confined to an in-memory, single-table analytical sandbox."""
from __future__ import annotations
import pandas as pd
import duckdb
import sqlglot
from sqlglot import exp

ROW_CAP = 1000
BLOCKED_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "sqlite_scan", "postgres_scan", "httpfs", "query_table", "glob",
    "getenv", "read_ndjson", "parquet_scan", "sqlite_attach", "http_get", "http_post",
}


def validate_select(sql: str) -> str:
    sql = (sql or "").strip()
    if not sql or len(sql) > 20_000:
        raise ValueError("Enter one SELECT query (maximum 20,000 characters).")
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"SQL syntax error: {e}") from e
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Only a single SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT queries are allowed (CTEs and UNION are supported).")
    banned = tuple(
        getattr(exp, name) for name in (
            "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command",
            "Copy", "Merge", "Attach", "Detach", "Pragma", "Grant", "Transaction",
        ) if hasattr(exp, name)
    )
    for node in tree.walk():
        if isinstance(node, banned):
            raise ValueError("Writing files, changing settings, and modifying data are not allowed.")
        if isinstance(node, exp.Func) and node.sql_name().lower() in BLOCKED_FUNCTIONS:
            raise ValueError("External read / system functions are not allowed.")
        if isinstance(node, exp.Anonymous) and node.name.lower() in BLOCKED_FUNCTIONS:
            raise ValueError("External read / system functions are not allowed.")
    ctes = {cte.alias.casefold() for cte in tree.find_all(exp.CTE)}
    for tbl in tree.find_all(exp.Table):
        if tbl.catalog or tbl.db or tbl.name.casefold() not in ({"data"} | ctes):
            raise ValueError("Queries may reference only the uploaded 'data' table and local CTEs.")
    if not list(tree.find_all(exp.Table)):
        # SELECT 1 is fine, but no unsafe table function is needed.
        pass
    return sql.rstrip("; \n")


def execute_select(df: pd.DataFrame, sql: str, limit: int = ROW_CAP) -> tuple[pd.DataFrame, bool]:
    query = validate_select(sql)
    conn = duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "false"})
    try:
        # Materialize the uploaded DataFrame before disabling external access:
        # registered pandas scans themselves count as external access in DuckDB.
        conn.register("_incoming", df)
        conn.execute("CREATE TEMP TABLE data AS SELECT * FROM _incoming")
        conn.unregister("_incoming")
        conn.execute("SET enable_external_access = false")
        # Guard displayed rows even when AI forgets LIMIT.
        safe = f"SELECT * FROM ({query}) AS __safe_result LIMIT {min(limit, ROW_CAP) + 1}"
        result = conn.execute(safe).fetchdf()
        truncated = len(result) > min(limit, ROW_CAP)
        return result.iloc[:min(limit, ROW_CAP)].copy(), truncated
    except duckdb.Error as e:
        raise ValueError(f"Query could not run: {e}") from e
    finally:
        conn.close()
