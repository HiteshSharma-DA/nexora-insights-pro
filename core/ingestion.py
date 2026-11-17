"""Defensive CSV loading and schema inference for user-uploaded datasets."""
from __future__ import annotations

from io import BytesIO
import re
import pandas as pd
import numpy as np

MAX_UPLOAD_MB = 50
MAX_ROWS = 350_000
MAX_COLS = 150


def unique_column_names(columns):
    names, seen = [], set()
    for i, value in enumerate(columns):
        base = str(value).strip() or f"column_{i + 1}"
        base = re.sub(r"\s+", " ", base)
        name, n = base, 2
        while name.lower() in seen:
            name, n = f"{base}_{n}", n + 1
        seen.add(name.lower())
        names.append(name)
    return names


def load_csv(raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    if not raw:
        raise ValueError("The CSV file is empty.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Maximum upload size is {MAX_UPLOAD_MB} MB for this deployment.")
    messages = []
    errors = []
    df = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        for sep in (None, ","):
            try:
                options = {"encoding": encoding, "nrows": MAX_ROWS + 1, "low_memory": False}
                if sep is None:
                    options.update({"sep": None, "engine": "python"})
                else:
                    options.update({"sep": sep})
                candidate = pd.read_csv(BytesIO(raw), **options)
                if candidate.shape[1] == 1 and b"," in raw[:4000] and sep is None:
                    continue
                df = candidate
                if encoding != "utf-8-sig":
                    messages.append(f"Decoded using {encoding} encoding.")
                break
            except (UnicodeError, pd.errors.ParserError, ValueError) as exc:
                errors.append(str(exc))
        if df is not None:
            break
    if df is None:
        raise ValueError("Unable to read this CSV. Confirm its delimiter and encoding. " + "; ".join(errors[-2:]))
    if df.shape[0] > MAX_ROWS:
        raise ValueError(f"File exceeds {MAX_ROWS:,} rows. Please filter or split it before uploading; no rows were silently discarded.")
    if df.shape[1] > MAX_COLS:
        raise ValueError(f"File exceeds {MAX_COLS} columns.")
    if df.empty or df.shape[1] == 0:
        raise ValueError("The CSV contains no data rows or columns.")
    df.columns = unique_column_names(df.columns)
    df = df.dropna(axis=1, how="all")
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        s = s.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA, "None": pd.NA})
        # Convert only very strongly numeric-looking columns; preserve IDs with leading zeros.
        nonmissing = s.dropna()
        if len(nonmissing) and len(nonmissing) <= MAX_ROWS:
            sample = nonmissing.iloc[:2000]
            looks_like_id = bool(re.search(r"(^|[_\W])(id|code|sku|zip|postal)([_\W]|$)", col, flags=re.I))
            leading_zero = sample.str.match(r"^0\d+$").any()
            vals = pd.to_numeric(sample.str.replace(r"[$£€,]", "", regex=True).str.replace("%", "", regex=False), errors="coerce")
            if not looks_like_id and not leading_zero and vals.notna().mean() >= 0.98:
                converted = pd.to_numeric(s.str.replace(r"[$£€,]", "", regex=True).str.replace("%", "", regex=False), errors="coerce")
                df[col] = converted
                messages.append(f"Converted '{col}' to a numeric column.")
            else:
                df[col] = s.astype(object).where(s.notna(), None)
    return df, messages


def prepare_dates(df: pd.DataFrame, date_col: str | None) -> tuple[pd.DataFrame, int]:
    if not date_col:
        return df, 0
    out = df.copy()
    parsed = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    invalid = int(parsed.isna().sum() - out[date_col].isna().sum())
    out[date_col] = parsed.dt.tz_convert(None)
    return out, max(0, invalid)


def quality_profile(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for name in df.columns:
        s = df[name]
        result.append({
            "column": name,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "null_%": round(float(s.isna().mean() * 100), 2),
            "distinct": int(s.nunique(dropna=True)),
            "example": str(s.dropna().iloc[0])[:70] if s.notna().any() else "",
        })
    return pd.DataFrame(result)
