"""NL -> guarded SQL; common questions work without a paid model."""
from __future__ import annotations
import json
import re
import pandas as pd
from .metadata import Mapping, qident, schema_summary
from .sql_engine import validate_select
from .diagnostics import locate_value


def local_query(question: str, df: pd.DataFrame, mapping: Mapping) -> tuple[str, str] | None:
    """Small, explicitly-supported natural-language templates (without AI or invented answers)."""
    q = question.casefold()
    metric = qident(mapping.measure) if mapping.measure else None
    dim = qident(mapping.dimension) if mapping.dimension else None
    category = qident(mapping.category) if mapping.category else None
    date = qident(mapping.date) if mapping.date else None

    if re.search(r"\b(how many rows|row count|number of rows|kitne rows|count records)\b", q):
        return "SELECT COUNT(*) AS total_rows FROM data", "The number of records in this dataset."
    if metric is None:
        return None
    if mapping.order_id and re.search(r"\b(unique orders|distinct orders|number of orders|how many orders)\b", q):
        return f"SELECT COUNT(DISTINCT {qident(mapping.order_id)}) AS distinct_orders FROM data", "Unique order identifiers counted."
    if date and any(t in q for t in ("monthly", "trend", "over time", "month", "time series", "mahine")):
        return (f"SELECT DATE_TRUNC('month', {date}) AS month, SUM({metric}) AS total_measure "
                f"FROM data WHERE {date} IS NOT NULL GROUP BY 1 ORDER BY 1",
                f"Monthly totals of {mapping.measure}.")
    from .diagnostics import locate_value
    segment = locate_value(question, df, mapping.dimension)
    if segment is not None and dim:
        value = "'" + segment.replace("'", "''") + "'"
        operator = "AVG" if re.search(r"\b(avg|average|mean)\b", q) else "SUM"
        return (f"SELECT CAST({dim} AS VARCHAR) AS segment, {operator}({metric}) AS total_measure, "
                f"COUNT(*) AS record_count FROM data "
                f"WHERE LOWER(CAST({dim} AS VARCHAR)) = LOWER({value}) GROUP BY 1",
                f"{operator} of {mapping.measure} for {segment}.")
    aggregation = "AVG" if re.search(r"\b(avg|average|mean)\b", q) else "SUM"
    group_col = None
    if category and any(t in q for t in ("product", "category", "channel", "item")):
        group_col = category
    elif dim and any(t in q for t in ("region", "segment", "area", "by ", "top", "bottom", "highest", "lowest", "best", "worst", "compare")):
        group_col = dim
    if group_col:
        match = re.search(r"\b(?:top|bottom|lowest|highest)\s+(\d{1,2})", q)
        n = max(1, min(int(match.group(1)), 25)) if match else 10
        order = "ASC" if any(t in q for t in ("bottom", "lowest", "least", "worst")) else "DESC"
        return (f"SELECT CAST({group_col} AS VARCHAR) AS segment, {aggregation}({metric}) AS total_measure, "
                f"COUNT(*) AS record_count FROM data WHERE {metric} IS NOT NULL GROUP BY 1 "
                f"ORDER BY total_measure {order} LIMIT {n}",
                f"{aggregation} of {mapping.measure} by {group_col.strip(chr(34))}.")
    if any(t in q for t in ("total", "sum", "overall", "kitni", "kitna", "sales", "revenue", "profit")):
        return f"SELECT {aggregation}({metric}) AS total_measure FROM data", f"{aggregation} of {mapping.measure} across the filtered dataset."
    return None


def generate_sql(question: str, df: pd.DataFrame, mapping: Mapping, api_key: str, model: str = "gpt-4.1-mini") -> tuple[str, str]:
    """Only schema/question are shared externally. No raw records or aggregates are sent."""
    if not api_key:
        raise ValueError("AI SQL generation needs an API key; built-in templates are free.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=25, max_retries=1)
    system = (
        "You translate business questions into DuckDB SELECT statements against one table named data. "
        "Return ONLY compact JSON with keys sql and explanation. SQL must be one read-only SELECT, "
        "use quoted identifiers, and never read files, external locations, URLs or functions, or modify state. "
        "Use only columns in the schema. For business questions, group and aggregate correctly. "
        "Prefer LIMIT 50 for detail queries. Do not invent columns or data. "
        "If a request is not answerable using these columns, return sql as an empty string and explain what column is missing."
    )
    payload = (
        "TABLE: data\nSCHEMA:\n" + schema_summary(df)
        + "\nUSER-CHOSEN ROLES: " + str(mapping)
        + "\nQUESTION: " + question[:1200]
    )
    response = client.responses.create(model=model, instructions=system, input=payload, max_output_tokens=600, store=False)
    raw = response.output_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("AI returned invalid SQL JSON. Please retry or use the SQL Workbench.") from exc
    sql = doc.get("sql") or ""
    if not sql:
        raise ValueError(doc.get("explanation") or "The requested information isn't available in the dataset.")
    return validate_select(sql), str(doc.get("explanation", "AI-generated query; verify its logic."))
