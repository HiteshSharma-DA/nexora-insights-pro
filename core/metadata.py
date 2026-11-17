"""Meaningful column defaults, while always allowing users to override mappings."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Mapping:
    measure: str | None = None
    dimension: str | None = None
    date: str | None = None
    category: str | None = None
    order_id: str | None = None


ROLE_PATTERNS = {
    "measure": ("sales", "revenue", "net_sales", "amount", "gmv", "total", "profit", "income", "price", "value"),
    "dimension": ("region", "zone", "territory", "area", "state", "market", "country", "segment", "branch"),
    "date": ("order_date", "transaction_date", "invoice_date", "date", "timestamp", "created_at", "month"),
    "category": ("category", "product_category", "sub_category", "product", "department", "channel", "type"),
    "order_id": ("order_id", "invoice_id", "transaction_id", "receipt_id", "order_number"),
}


def guess_role(df: pd.DataFrame, role: str) -> str | None:
    if role == "measure":
        candidates = list(df.select_dtypes(include="number").columns)
    elif role in ("date", "order_id"):
        candidates = list(df.columns)
    else:
        candidates = [c for c in df.columns if df[c].nunique(dropna=True) <= 200]
    if not candidates:
        return None
    patterns = ROLE_PATTERNS[role]
    for p in patterns:
        for c in candidates:
            if c.casefold().replace(" ", "_") == p:
                return c
    for p in patterns:
        for c in candidates:
            if p in c.casefold().replace(" ", "_"):
                return c
    if role == "measure":
        return candidates[0]
    if role == "dimension":
        cats = [c for c in candidates if 1 < df[c].nunique(dropna=True) <= 30 and not pd.api.types.is_numeric_dtype(df[c])]
        return cats[0] if cats else None
    if role == "date":
        dates = [c for c in candidates if pd.api.types.is_datetime64_any_dtype(df[c])]
        return dates[0] if dates else None
    return None


def guess_mapping(df: pd.DataFrame) -> Mapping:
    return Mapping(**{key: guess_role(df, key) for key in ROLE_PATTERNS})


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def schema_summary(df: pd.DataFrame) -> str:
    return "\n".join(f"- {qident(col)}: {str(df[col].dtype)}" for col in df.columns)
