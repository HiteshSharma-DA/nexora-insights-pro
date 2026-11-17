"""Evidence-first analysis. Low level and falling trend are deliberately different intents."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import pandas as pd
from .metadata import Mapping, qident
from .sql_engine import execute_select


@dataclass
class Diagnosis:
    headline: str
    explanation: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    queries: list[tuple[str, str, pd.DataFrame]] = field(default_factory=list)
    caveat: str = (
        "Observed gaps and changes are not proven causes. Row counts are not necessarily order counts. "
        "Validate data coverage, prices, product mix, distribution and other missing business context before acting."
    )
    mode: str = "level"
    segment: str = ""
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    drivers: pd.DataFrame = field(default_factory=pd.DataFrame)


def _lit(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _num(n: float) -> str:
    return f"{float(n):,.2f}"


def _delta(n: float, old: float) -> str:
    return f"{(n - old) / abs(old) * 100:+.1f}%" if old else "not comparable (zero baseline)"


def question_intent(question: str) -> str:
    """Trend requires time/decline language; 'lower than peers' means level."""
    q = " " + question.casefold() + " "
    trend = (
        r"\b(declin\w*|drop\w*|fall\w*|fell|chang\w*|decreas\w*|down|"
        r"downward|down by|down since|last month|last week|compared to last|"
        r"past month|previous month|over time|trend|dip|shrunk|reduc\w*)\b"
    )
    if re.search(trend, q) or any(s in q for s in ("गिरावट", "घट रही", "घट गया", "पिछले महीने", "क्यों गिरी")):
        return "trend"
    return "level"


def locate_value(question: str, df: pd.DataFrame, dimension: str | None) -> str | None:
    if not dimension or dimension not in df:
        return None
    text = " " + question.casefold() + " "
    vals = df[dimension].dropna().astype(str).unique()
    for candidate in sorted(vals, key=lambda s: -len(s))[:250]:
        if re.search(r"(?<!\w)" + re.escape(candidate.casefold()) + r"(?!\w)", text):
            return candidate
    return None


def diagnose_segment(df: pd.DataFrame, mapping: Mapping, value: str,
                     days: int = 30, mode: str = "trend") -> Diagnosis:
    if not mapping.measure or not mapping.dimension:
        raise ValueError("Map a numeric measure and group (for example, Revenue and Region).")
    if mode == "trend" and mapping.date and pd.to_datetime(df[mapping.date], errors="coerce").notna().sum() >= 3:
        return _trend(df, mapping, value, days)
    return _level(df, mapping, value)


def _level(df: pd.DataFrame, mapping: Mapping, value: str) -> Diagnosis:
    measure, dim = qident(mapping.measure), qident(mapping.dimension)
    report = Diagnosis(f"Why is {value} {mapping.measure} different? · Region comparison", mode="level", segment=value)
    query = f'''SELECT CAST({dim} AS VARCHAR) AS segment,
        COUNT(*) AS rows_count, SUM({measure}) AS total_measure,
        AVG({measure}) AS avg_per_row
      FROM data WHERE {measure} IS NOT NULL AND {dim} IS NOT NULL
      GROUP BY 1 ORDER BY total_measure DESC'''
    summary, _ = execute_select(df, query)
    report.summary = summary
    report.queries.append(("All-time segment comparison · executed SQL", query, summary))
    matched = summary[summary["segment"].astype(str).str.casefold() == value.casefold()]
    if matched.empty:
        report.explanation.append(f"No measured rows were found for {value}. Choose another group or check the mapped columns.")
        return report
    current = matched.iloc[0]
    rows, total, average = int(current["rows_count"]), float(current["total_measure"]), float(current["avg_per_row"])
    report.explanation.append(
        f"**Full available dataset (not a 30-day decline):** {value} has **{_num(total)}** total {mapping.measure} "
        f"across **{rows:,} recorded rows**, averaging **{_num(average)} per row**."
    )
    peers = summary[summary["segment"].astype(str).str.casefold() != value.casefold()]
    if peers.empty:
        report.explanation.append("There are no other groups in the current data view to compare against.")
    else:
        peer_total = float(peers["total_measure"].mean())
        peer_avg = float(peers["avg_per_row"].mean())
        peer_count = float(peers["rows_count"].mean())
        diff_t = _delta(total, peer_total)
        report.explanation.append(
            f"**Peer benchmark:** the other {len(peers)} groups average **{_num(peer_total)} total** "
            f"({diff_t} for {value}); their average revenue per row is **{_num(peer_avg)}**, "
            f"vs **{_num(average)}** in {value}."
        )
        report.explanation.append(
            f"**Volume vs value:** {value} has **{rows:,} rows** versus peers' **{peer_count:,.1f} rows on average**. "
            f"Both transaction volume and value per row should be assessed separately."
        )
        if len(summary) > 1 and total > float(summary["total_measure"].min()) + 1e-8:
            lowest = summary.iloc[-1]["segment"]
            report.explanation.append(f"**Accuracy check:** {value} is not the lowest total in this data; **{lowest}** is.")
        if rows < peer_count * .90:
            report.actions.append("Check whether this group has fewer stores, active customers, leads or transaction opportunities; request coverage data before assuming weaker demand.")
        if average < peer_avg * .95:
            report.actions.append("Investigate lower value per recorded row: compare product mix, quantities, unit prices and discounts with similar groups.")
        if len(peers) and rows >= peer_count and average < peer_avg:
            report.actions.append("Volume is not obviously the main gap; prioritise product/price mix and average-value comparisons rather than only increasing transaction count.")
        if mapping.order_id:
            quoted_id = qident(mapping.order_id)
            order_sql = f"SELECT CAST({dim} AS VARCHAR) AS segment, COUNT(DISTINCT {quoted_id}) AS unique_orders FROM data GROUP BY 1 ORDER BY 2 DESC"
            orders, _ = execute_select(df, order_sql)
            report.queries.append(("Unique orders by group (if IDs are populated)", order_sql, orders))
            order_row = orders[orders["segment"].astype(str).str.casefold() == value.casefold()]
            if not order_row.empty:
                report.explanation.append(f"**Distinct IDs:** {value} has **{int(order_row.iloc[0]['unique_orders']):,} unique {mapping.order_id} values** (assuming each ID represents one order).")
        if mapping.category and mapping.category != mapping.dimension:
            cat = qident(mapping.category)
            sql = f'''WITH grouped AS (
              SELECT CAST({dim} AS VARCHAR) AS segment,
                COALESCE(CAST({cat} AS VARCHAR), '(Missing)') AS driver,
                COUNT(*) AS rows_count, SUM({measure}) AS total_measure,
                AVG({measure}) AS avg_per_row
              FROM data WHERE {measure} IS NOT NULL AND {dim} IS NOT NULL
              GROUP BY 1, 2
            ), target AS (
              SELECT * FROM grouped WHERE LOWER(segment) = LOWER({_lit(value)})
            ), peers AS (
              SELECT driver, AVG(total_measure) AS peer_mean_total,
                AVG(rows_count) AS peer_mean_rows, COUNT(*) AS peer_groups_present
              FROM grouped WHERE LOWER(segment) != LOWER({_lit(value)}) GROUP BY driver
            )
            SELECT COALESCE(target.driver, peers.driver) AS driver,
              COALESCE(target.total_measure, 0) AS target_total,
              COALESCE(target.rows_count, 0) AS target_rows,
              COALESCE(peers.peer_mean_total, 0) AS peer_mean_total,
              COALESCE(target.total_measure, 0) - COALESCE(peers.peer_mean_total, 0) AS difference,
              COALESCE(peers.peer_groups_present, 0) AS peer_groups_present
            FROM target FULL OUTER JOIN peers ON target.driver = peers.driver
            ORDER BY difference ASC LIMIT 50'''
            drivers, _ = execute_select(df, sql)
            report.drivers = drivers
            report.queries.append((f"{mapping.category} gap vs other groups · executed SQL", sql, drivers))
            if not drivers.empty:
                worst = drivers.iloc[0]
                report.explanation.append(
                    f"**Biggest {mapping.category} gap vs peers:** **{worst['driver']}**, "
                    f"{_num(float(worst['target_total']))} in {value} vs {_num(float(worst['peer_mean_total']))} "
                    f"average among peer groups where present. This is a category comparison, not causal proof."
                )
                report.actions.append(f"Review **{worst['driver']}** distribution, inventory availability, conversion and pricing; validate differences across comparable locations before running an experiment.")
    if not report.actions:
        report.actions.append("Check category mix, conversion rate and average order value against comparable regions; collect missing denominators (customers, stores, traffic).")
    report.actions.append("Assign an owner and test one targeted change for 2–4 weeks with a measurable KPI and comparison/control group where feasible.")
    return report


def _trend(df: pd.DataFrame, mapping: Mapping, value: str, days: int) -> Diagnosis:
    dim, measure = qident(mapping.dimension), qident(mapping.measure)
    date = mapping.date
    days = max(1, min(int(days), 365))
    finish = pd.to_datetime(df[date], errors="coerce").dropna().max().normalize()
    start = finish - pd.Timedelta(days=days - 1)
    prev_end = start - pd.Timedelta(days=1)
    prev_start = start - pd.Timedelta(days=days)
    dt = qident(date)
    cond_now = f"CAST({dt} AS DATE) BETWEEN DATE '{start.date()}' AND DATE '{finish.date()}'"
    cond_old = f"CAST({dt} AS DATE) BETWEEN DATE '{prev_start.date()}' AND DATE '{prev_end.date()}'"
    sql = f'''SELECT CAST({dim} AS VARCHAR) AS segment,
      CASE WHEN {cond_now} THEN 'Current' ELSE 'Previous' END AS period,
      COUNT(*) AS rows_count, SUM({measure}) AS total_measure,
      AVG({measure}) AS avg_per_row
    FROM data WHERE ({cond_now} OR {cond_old}) AND {measure} IS NOT NULL
    GROUP BY 1, 2 ORDER BY 1, 2'''
    summary, _ = execute_select(df, sql)
    report = Diagnosis(f"What changed in {value}? · {mapping.measure}", mode="trend", segment=value, summary=summary)
    report.queries.append(("Equal-length periods · executed SQL", sql, summary))
    seg = summary[summary["segment"].astype(str).str.casefold() == value.casefold()]
    latest = seg[seg["period"] == "Current"]
    previous = seg[seg["period"] == "Previous"]
    current_count = int(latest.iloc[0]["rows_count"]) if not latest.empty else 0
    old_count = int(previous.iloc[0]["rows_count"]) if not previous.empty else 0
    current = float(latest.iloc[0]["total_measure"]) if not latest.empty else 0.0
    old = float(previous.iloc[0]["total_measure"]) if not previous.empty else 0.0
    report.explanation.append(f"**Comparable windows:** {start.date()}–{finish.date()} versus {prev_start.date()}–{prev_end.date()} ({days} days each, ending at the latest recorded date).")
    report.explanation.append(f"**Observed {mapping.measure}:** {_num(current)} across {current_count:,} rows recently; {_num(old)} across {old_count:,} rows previously.")
    if not current_count or not old_count:
        report.explanation.append("**Insufficient period coverage:** at least one window has no records for this group. A 0% or -100% sales change must not be interpreted as a confirmed business decline; it may indicate gaps or sparse data.")
        report.actions.append("Verify that both periods have comparable data coverage and all stores/channels have reported before calculating a decline.")
    else:
        report.explanation.append(f"**Measured change:** {_delta(current, old)} in total, {_delta(current_count, old_count)} in recorded rows, and {_delta(current/current_count, old/old_count)} in value per row.")
        if current_count < old_count * .9:
            report.actions.append("Investigate fewer recorded transactions: validate completeness, customer traffic, store coverage and conversion before changing campaigns.")
        if current/current_count < old/old_count * .95:
            report.actions.append("Review average value per row: product mix, quantities, discounts and returns could matter—compare them with earlier data.")
    if mapping.category and mapping.category != mapping.dimension and old_count and current_count:
        cat = qident(mapping.category)
        sql_cat = f'''SELECT COALESCE(CAST({cat} AS VARCHAR), '(Missing)') AS driver,
          SUM(CASE WHEN {cond_now} THEN {measure} ELSE 0 END) AS current_value,
          SUM(CASE WHEN {cond_old} THEN {measure} ELSE 0 END) AS previous_value,
          SUM(CASE WHEN {cond_now} THEN {measure} ELSE 0 END) -
          SUM(CASE WHEN {cond_old} THEN {measure} ELSE 0 END) AS delta
        FROM data WHERE LOWER(CAST({dim} AS VARCHAR))=LOWER({_lit(value)})
          AND ({cond_now} OR {cond_old}) AND {measure} IS NOT NULL
        GROUP BY 1 ORDER BY delta ASC LIMIT 50'''
        drivers, _ = execute_select(df, sql_cat)
        report.drivers = drivers
        report.queries.append((f"{mapping.category} contribution to period change · executed SQL", sql_cat, drivers))
        if not drivers.empty and float(drivers.iloc[0]["delta"]) < 0:
            worst = drivers.iloc[0]
            report.explanation.append(f"**Largest negative {mapping.category} contribution:** {worst['driver']} ({_num(float(worst['delta']))} change). This is a decomposition, not a proven cause.")
            report.actions.append(f"Investigate {worst['driver']} availability, prices and customer/channel mix across both periods.")
    if not report.actions:
        report.actions.append("Compare channel, pricing, quantity and product metrics across equivalent periods before changing operations.")
    report.actions.append("Run a 2–4 week controlled improvement experiment with a named owner and pre-defined success metric.")
    report.caveat += " The latest available date may reflect a partial reporting window; confirm coverage."
    return report
