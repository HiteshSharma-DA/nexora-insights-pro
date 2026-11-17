"""NEXORA INSIGHTS — business-first, evidence-led data intelligence by Hitesh Sharma."""
from __future__ import annotations

import html
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.agent import generate_sql, local_query
from core.charting import COLORS, anomaly_chart, by_dimension, correlation_heatmap, polish, time_trend
from core.diagnostics import diagnose_segment, locate_value, question_intent
from core.forecasting import aggregate_series, flag_anomalies, forecast_with_backtest
from core.ingestion import load_csv, prepare_dates, quality_profile
from core.metadata import Mapping, guess_mapping
from core.reports import build_html_report, build_pdf_report
from core.sql_engine import execute_select
from core.statistics import chi_square_test, outlier_profile, summary_stats, welch_test

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="NEXORA INSIGHTS | Decision Intelligence by Hitesh Sharma",
    page_icon="◆", layout="wide", initial_sidebar_state="expanded",
)
st.markdown("<style>" + (ROOT / "assets" / "style.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_dataframe(data: bytes):
    return load_csv(data)


def safe(s: object) -> str:
    return html.escape(str(s), quote=True)


def format_number(n: float | int | None, prefix: str = "") -> str:
    if n is None or pd.isna(n):
        return "—"
    a = abs(float(n))
    if a >= 1_000_000_000:
        value = f"{n/1_000_000_000:,.2f}B"
    elif a >= 1_000_000:
        value = f"{n/1_000_000:,.2f}M"
    elif a >= 1000:
        value = f"{n/1000:,.1f}K"
    else:
        value = f"{n:,.2f}" if not float(n).is_integer() else f"{n:,.0f}"
    return f"{prefix}{value}"


def div_label(label: str, title: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="section-head"><span>{safe(label)}</span><h2>{safe(title)}</h2>'
        f'<p>{safe(detail)}</p></div>', unsafe_allow_html=True,
    )


def tiny_card(label: str, value: str, foot: str = "", accent: str = "red") -> None:
    st.markdown(
        f'<div class="tiny-card {safe(accent)}"><div class="tiny-label">{safe(label)}</div>'
        f'<div class="tiny-value">{safe(value)}</div><div class="tiny-foot">{safe(foot)}</div></div>',
        unsafe_allow_html=True,
    )


def section_note(message: str, icon: str = "ℹ") -> None:
    st.markdown(f'<div class="info-note"><b>{safe(icon)}</b> {safe(message)}</div>', unsafe_allow_html=True)


def result_table(df: pd.DataFrame, caption: str = "") -> None:
    if caption:
        st.caption(caption)
    st.dataframe(df, use_container_width=True, hide_index=True)


def query_block(label: str, sql: str, result: pd.DataFrame | None = None) -> None:
    with st.expander(f"Show SQL evidence · {label}", expanded=False):
        st.code(sql, language="sql")
        if result is not None:
            result_table(result.head(100), "Results from the uploaded dataset (up to 100 rows shown)")
        st.download_button("Download SQL", sql.encode("utf-8"), file_name="analysis_query.sql", mime="text/plain", key=f"sql_{abs(hash((label, sql))) % 10**12}")


def display_diagnosis(diag) -> None:
    st.markdown(
        f'<div class="diagnosis-banner"><b>{safe(diag.headline)}</b>'
        f'<small>{"Equal-length time-window investigation" if diag.mode == "trend" else "Whole-dataset comparison against peer groups"} · SQL-verified evidence</small></div>',
        unsafe_allow_html=True,
    )
    for line in diag.explanation:
        st.markdown("- " + line)
    if diag.mode == "level" and not diag.summary.empty:
        st.markdown("#### Compare business groups")
        chart = px.bar(diag.summary.sort_values("total_measure"), x="total_measure", y="segment", orientation="h",
                       color_discrete_sequence=COLORS, text="total_measure")
        chart.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(polish(chart), use_container_width=True)
        st.caption("Total value, row counts and average per row answer different questions. A low total is not a proven downward trend.")
    if not diag.drivers.empty:
        with st.expander("Explore category/product evidence", expanded=True):
            result_table(diag.drivers.head(30))
            if "difference" in diag.drivers.columns:
                fig = px.bar(diag.drivers.head(12).sort_values("difference"), x="difference", y="driver",
                             orientation="h", color_discrete_sequence=COLORS)
                fig.update_layout(xaxis_title="Difference vs peer category average", yaxis_title="")
                st.plotly_chart(polish(fig), use_container_width=True)
    st.markdown("#### Recommended next steps")
    for index, action in enumerate(diag.actions, 1):
        st.markdown(f"**{index}.** {action}")
    st.caption("Important: " + diag.caveat)
    st.markdown("#### Show the evidence · Executed SQL")
    for label, sql, results in diag.queries:
        query_block(label, sql, results)


def display_result(answer: dict) -> None:
    if answer.get("diagnosis"):
        display_diagnosis(answer["diagnosis"])
        return
    st.markdown(f"**Finding:** {answer['explanation']}")
    df_res = answer["result"]
    if df_res.empty:
        st.info("This question returned no matching records. Try different filters or a broader question.")
    else:
        if {"segment", "total_measure"}.issubset(df_res.columns) and not df_res["total_measure"].isna().all():
            largest = df_res.loc[pd.to_numeric(df_res["total_measure"], errors="coerce").idxmax()]
            st.success(f"Among the returned groups, **{largest['segment']}** has the highest value: **{format_number(largest['total_measure'])}**.")
        elif {"month", "total_measure"}.issubset(df_res.columns) and len(df_res) >= 2:
            old, new = pd.to_numeric(df_res["total_measure"], errors="coerce").iloc[[0, -1]]
            if pd.notna(old) and pd.notna(new):
                st.info(f"First period: {format_number(old)} → last period: {format_number(new)}. Compare equivalent seasons before attributing the change to any cause.")
        if len(df_res) == 1 and len(df_res.columns) == 1:
            val = df_res.iat[0, 0]
            if isinstance(val, (int, float, np.number)):
                st.metric("Result", format_number(val))
        result_table(df_res)
        if answer.get("truncated"):
            st.warning("Results limited to 1,000 rows. Add filters or an aggregation to see the relevant results.")
        st.caption("The answer uses your uploaded data. Review column mappings and SQL before business decisions.")
        st.download_button("Download answer as CSV", df_res.to_csv(index=False).encode("utf-8"),
                           file_name="question_result.csv", mime="text/csv", key="qa_result_download")
    query_block("Verified query", answer["sql"], None)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="brand-block"><div class="brand-mark">N</div>'
        '<div><div class="brand-title">NEXORA <b>INSIGHTS</b></div>'
        '<div class="brand-sub">DECISION INTELLIGENCE</div></div></div>', unsafe_allow_html=True,
    )
    st.sidebar.markdown('<p class="side-credit">An analytics product by <strong>Hitesh Sharma</strong></p>', unsafe_allow_html=True)


def mapping_select(label: str, options: list[str], proposed: str | None, key: str, help_text: str) -> str | None:
    available = ["— Not available —"] + options
    idx = available.index(proposed) if proposed in available else 0
    value = st.selectbox(label, available, index=idx, key=key, help=help_text)
    return None if value == available[0] else value


def sidebar_data() -> tuple[pd.DataFrame | None, Mapping | None, str, list[str]]:
    sidebar_brand()
    st.sidebar.markdown("<div class='sidebar-step'>01 · YOUR DATA</div>", unsafe_allow_html=True)
    upload = st.sidebar.file_uploader("Upload your CSV", type=["csv"], help="Your data stays in this Streamlit session except if you choose optional external AI. Max 50 MB.")
    demo = (ROOT / "demo" / "retail_sales_demo.csv")
    if upload is not None:
        raw, label = upload.getvalue(), upload.name
        st.sidebar.success("CSV uploaded")
    else:
        raw, label = demo.read_bytes(), "Example retail sales · SAMPLE DATA"
        st.sidebar.caption("Showing example data. Upload a CSV to replace it.")
    file_id = hashlib.sha256(raw).hexdigest()[:16]
    if st.session_state.get("current_file_id") != file_id:
        st.session_state["current_file_id"] = file_id
        st.session_state.pop("last_answer", None)
        st.session_state.pop("chosen_diag", None)
        st.session_state.pop("last_diagnosed_group", None)
        st.session_state.pop("last_diagnosis_mode", None)
    try:
        df, messages = get_dataframe(raw)
    except Exception as exc:
        st.sidebar.error(f"Could not open CSV: {exc}")
        return None, None, label, []
    st.sidebar.markdown("<div class='sidebar-step'>02 · UNDERSTAND YOUR COLUMNS</div>", unsafe_allow_html=True)
    st.sidebar.caption("We guessed these automatically. Change them if needed.")
    guess = guess_mapping(df)
    nums = list(df.select_dtypes(include="number").columns)
    all_cols = list(df.columns)
    low_card = [c for c in all_cols if 1 < df[c].nunique(dropna=True) <= 200]
    # Key changes after a different file is uploaded, preventing stale selection state.
    signature = file_id
    with st.sidebar.expander("Adjust column meanings", expanded=False):
        measure = mapping_select("Main number (sales, revenue, etc.)", nums, guess.measure, f"measure_{signature}", "Which numeric column should be added or compared?")
        dimension = mapping_select("Business group (region, team, etc.)", low_card, guess.dimension, f"dim_{signature}", "What are you comparing?")
        date = mapping_select("Date", all_cols, guess.date, f"date_{signature}", "Needed for trends and before/after comparisons.")
        category = mapping_select("Secondary group (product, category)", low_card, guess.category, f"cat_{signature}", "Used to investigate which products contribute to a change.")
        order_id = mapping_select("Order identifier (optional)", all_cols, guess.order_id, f"id_{signature}", "Used for order counting, not required.")
    mapping = Mapping(measure, dimension, date, category, order_id)
    if date:
        df, invalid_count = prepare_dates(df, date)
        if invalid_count:
            st.sidebar.warning(f"{invalid_count:,} date values were invalid and treated as missing.")
    st.sidebar.markdown("<div class='sidebar-step'>03 · FOCUS THE ANALYSIS</div>", unsafe_allow_html=True)
    chosen: list[str] = []
    if dimension and df[dimension].nunique(dropna=True) <= 100:
        values = sorted(str(v) for v in df[dimension].dropna().unique())
        chosen = st.sidebar.multiselect("Show only these groups (optional)", values, default=[], key=f"filter_{signature}_{dimension}")
        if chosen:
            df = df[df[dimension].astype(str).isin(chosen)].copy()
    # Invalidate answers whenever mappings or filters change, not just new uploads.
    context = (file_id, mapping, tuple(chosen))
    if st.session_state.get("analysis_context") != context:
        st.session_state["analysis_context"] = context
        st.session_state.pop("last_answer", None)
        st.session_state.pop("sql_result", None)
    st.sidebar.caption(f"Analysing **{len(df):,}** records · **{len(df.columns)}** columns")
    with st.sidebar.expander("Privacy & limitations"):
        st.caption("Uploads are processed in this app session. Optional AI sends your question and column names/types to the configured AI provider, not raw rows. Do not upload sensitive customer data to a public deployment without access controls. Statistical comparisons do not prove causation.")
    return df, mapping, label, messages


def render_hero(label: str, rows: int, demo: bool) -> None:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">◆ NEXORA / INTELLIGENCE AT WORK</div>'
        '<h1>Turn your data into <em>decisions.</em></h1>'
        '<p>A beautifully simple analytics workspace. Upload your CSV, uncover meaningful insights, '
        'compare performance, and know what to do next.</p>'
        f'<div class="hero-tags"><span>{"DEMO DATA" if demo else "YOUR UPLOAD"}</span><span>{safe(label)}</span><span>{rows:,} records ready</span><span>By Hitesh Sharma</span></div>'
        '</div>', unsafe_allow_html=True,
    )


def overview(df: pd.DataFrame, mapping: Mapping) -> None:
    st.markdown(
        '<div class="how-grid">'
        '<div><span>01</span><strong>Upload your CSV</strong><small>Drag your file into the left panel.</small></div>'
        '<div><span>02</span><strong>Find instant answers</strong><small>Ask in simple words or open a chart.</small></div>'
        '<div><span>03</span><strong>Take action</strong><small>Investigate trends and export a report.</small></div>'
        '</div>', unsafe_allow_html=True,
    )
    div_label("YOUR BUSINESS AT A GLANCE", "Overview dashboard", "Everything important, without a single SQL query.")
    metric, dim, date = mapping.measure, mapping.dimension, mapping.date
    cols = st.columns(4, gap="medium")
    metrics = [
        ("Total records", format_number(len(df)), "Rows currently in view", "red"),
        (f"Total {metric}" if metric else "Columns available", format_number(df[metric].sum()) if metric else str(len(df.columns)), "Sum of all non-missing values" if metric else "No numerical metric mapped", "teal"),
        ("Distinct groups", format_number(df[dim].nunique()) if dim else "—", f"By {dim}" if dim else "Select a group in the sidebar", "violet"),
        ("Data completeness", f"{(1-df.isna().to_numpy().mean())*100:.1f}%", "Non-missing cells in this view", "orange"),
    ]
    for col, (label, value, foot, accent) in zip(cols, metrics):
        with col:
            tiny_card(label, value, foot, accent)
    st.write("")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.markdown("#### Performance over time")
        if metric and date and pd.api.types.is_datetime64_any_dtype(df[date]) and df[date].notna().sum() > 1:
            fig = time_trend(df, date, metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                last_date = pd.to_datetime(df[date], errors="coerce").max()
                if pd.notna(last_date) and last_date.normalize() < (last_date + pd.offsets.MonthEnd(0)).normalize():
                    st.caption("Latest month may be incomplete. Do not compare it directly with full months; forecasting excludes partial boundary months.")
            else:
                st.info("Not enough dated rows to draw a trend.")
        else:
            st.info("Map a numeric measure and date in the sidebar to see your trend.")
    with right:
        st.markdown("#### Top business segments")
        if metric and dim:
            fig, table = by_dimension(df, metric, dim, top_n=8)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Map a number and group to compare performance.")
    if metric and dim:
        st.markdown("#### Auto-generated observations")
        grp = df.groupby(dim, dropna=True, observed=True)[metric].sum().sort_values(ascending=False)
        if len(grp):
            best = grp.index[0]
            share = (grp.iloc[0] / grp.sum() * 100) if grp.sum() != 0 else None
            st.markdown(f"**Highest total:** **{best}** contributed **{format_number(grp.iloc[0])}**" + (f" ({share:.1f}% of total)." if share is not None else "."))
        if len(grp) > 1:
            last = grp.index[-1]
            st.markdown(f"**Lowest total:** **{last}** contributed **{format_number(grp.iloc[-1])}**. This does not necessarily imply a decline; compare similar-size segments and time periods.")
            st.button(f"Investigate {last}", type="secondary",
                      on_click=lambda target=str(last): st.session_state.update({"nav": "Why did it change?", "chosen_diag": target, "diag_default_mode": "level"}))
    with st.expander("Preview the uploaded data"):
        result_table(df.head(100), "First 100 records")
    if date and metric:
        section_note("Trends are based on recorded transactions. To identify causes, compare periods and investigate pricing, demand, channels and product mix.")


def questions(df: pd.DataFrame, mapping: Mapping) -> None:
    div_label("ASK IN PLAIN LANGUAGE", "Ask your data", "Simple questions produce data-backed answers. SQL is available if you want the technical proof.")
    examples = [
        (f"Why are {'North' if 'North' in set(df[mapping.dimension].dropna().astype(str)) else str(df[mapping.dimension].dropna().iloc[0])} {mapping.measure or 'revenue'} low?"
         if mapping.dimension and df[mapping.dimension].notna().any() else "Why is this segment performing poorly?"),
        f"Show top 5 {mapping.dimension or 'groups'} by {mapping.measure or 'revenue'}", f"What is total {mapping.measure or 'revenue'}?", f"Show monthly {mapping.measure or 'revenue'} trend",
    ]
    st.caption("Try a question:")
    cc = st.columns([1.5, 1, 1, 1], gap="small")
    for col, phrase in zip(cc, examples):
        with col:
            if st.button(phrase, use_container_width=True, key=f"example_{abs(hash(phrase))%10**9}"):
                st.session_state["ask_question"] = phrase
    server_key = os.environ.get("OPENAI_API_KEY", "")
    if not server_key:
        try:
            server_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    if server_key:
        section_note("Smart AI questions are enabled by the app owner. You don't need to enter an API key.", "✦")
    with st.expander("Optional AI settings · for advanced questions", expanded=False):
        st.caption("AI converts your question into SQL. The app validates it before executing against your upload. Only your question and column names/types are sent to the AI service; no records. API usage may incur charges.")
        entered_key = "" if server_key else st.text_input("OpenAI API key (optional)", type="password", key="user_api_key", placeholder="sk-…")
        key = server_key or entered_key
        model = st.selectbox("AI model", ["gpt-4.1-mini", "gpt-4o-mini"], key="ai_model")
        use_ai = st.checkbox("Use AI for open-ended SQL questions", value=bool(server_key), key="use_ai")
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input("What would you like to know?", key="ask_question", placeholder="e.g. Why is North region selling less?", max_chars=600)
        send = st.form_submit_button("Analyse my question", type="primary", use_container_width=True)
    if send:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                val = locate_value(question, df, mapping.dimension)
                why = any(s in question.casefold() for s in ("why", "reason", "decline", "drop", "low", "less", "lowest", "lower", "underperform", "behind", "gap", "poor", "weaker", "kyu", "kyon", "kam", "ghat", "क्यों", "कम", "गिर"))
                if why and val and mapping.measure and mapping.dimension:
                    analysis_mode = question_intent(question)
                    diagnosis = diagnose_segment(df, mapping, val, mode=analysis_mode)
                    st.session_state["last_diagnosed_group"] = val
                    st.session_state["last_diagnosis_mode"] = analysis_mode
                    st.session_state["last_answer"] = {"diagnosis": diagnosis}
                else:
                    if use_ai and key:
                        sql, explanation = generate_sql(question, df, mapping, key, model)
                    else:
                        found = local_query(question, df, mapping)
                        if not found:
                            st.session_state["last_answer"] = None
                            st.info("This question needs more context or optional AI. Try a suggested question, select a segment for 'why' analysis, or enable AI SQL generation.")
                            return
                        sql, explanation = found
                    result, truncated = execute_select(df, sql)
                    st.session_state["last_answer"] = {"result": result, "sql": sql, "explanation": explanation, "truncated": truncated}
            except Exception as exc:
                st.session_state["last_answer"] = None
                st.error(f"Could not analyse this question: {exc}")
    if st.session_state.get("last_answer"):
        st.markdown("---")
        display_result(st.session_state["last_answer"])
    else:
        section_note("For 'why' questions, map a region/group, sales number and date. The app will show comparisons, SQL evidence and action ideas.")


def diagnostics_page(df: pd.DataFrame, mapping: Mapping) -> None:
    div_label("UNDERSTAND THE DIFFERENCE", "Performance investigation", "A low result and a declining trend are different questions. Choose which one to investigate.")
    if not mapping.dimension or not mapping.measure:
        st.warning("Choose a numeric measure and a business group in the sidebar first.")
        return
    values = sorted(str(x) for x in df[mapping.dimension].dropna().unique())
    if not values:
        st.info("There are no groups in this view.")
        return
    chosen = st.session_state.pop("chosen_diag", None)
    idx = values.index(chosen) if chosen in values else (values.index(st.session_state.get("last_diagnosed_group")) if st.session_state.get("last_diagnosed_group") in values else 0)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        segment = st.selectbox(f"Which {mapping.dimension} do you want to investigate?", values, index=idx)
    with col2:
        options = ["Compared with other groups (whole dataset)", "Changed over time (period vs previous period)"]
        default_mode = st.session_state.pop("diag_default_mode", st.session_state.get("last_diagnosis_mode", "level"))
        purpose = st.selectbox("What is the question?", options, index=1 if default_mode == "trend" else 0)
    mode = "trend" if purpose == options[1] else "level"
    days = 30
    if mode == "trend":
        days = st.selectbox("Compare equal-length periods", [30, 7, 60, 90], format_func=lambda n: f"Last {n} recorded days vs previous {n} days")
    if st.button("Find evidence and next steps", type="primary", use_container_width=True):
        try:
            diagnosis = diagnose_segment(df, mapping, segment, days=days, mode=mode)
            st.session_state["last_diagnosed_group"] = segment
            st.session_state["last_diagnosis_mode"] = mode
            display_diagnosis(diagnosis)
        except Exception as exc:
            st.error(f"Could not run diagnosis: {exc}")
    section_note("This tool measures differences, shows executed SQL and recommends checks or experiments. It does not claim to know unobserved business causes.")


def explorer(df: pd.DataFrame, mapping: Mapping) -> None:
    div_label("MAKE IT VISUAL", "Interactive chart builder", "No formulas required. Pick columns, choose a chart and share what you discover.")
    num = list(df.select_dtypes(include="number").columns)
    cat = [c for c in df.columns if 1 < df[c].nunique(dropna=True) <= 500]
    if not num:
        st.warning("There are no numeric columns available for charts.")
        return
    a, b, c, d = st.columns(4)
    with a:
        kind = st.selectbox("Chart", ["Bar", "Line", "Pie", "Treemap", "Histogram", "Box", "Scatter", "Correlation"])
    with b:
        metric = st.selectbox("Value", num, index=num.index(mapping.measure) if mapping.measure in num else 0)
    with c:
        chosen = mapping.dimension if mapping.dimension in cat else (cat[0] if cat else None)
        group = st.selectbox("Group or X axis", cat if cat else ["(none)"], index=cat.index(chosen) if chosen else 0)
    with d:
        agg = st.selectbox("How to calculate", ["Sum", "Mean", "Count"])
    try:
        if kind in ("Bar", "Line", "Pie", "Treemap"):
            if not cat:
                st.info("Add a categorical field to use this chart.")
                return
            fig, tab = by_dimension(df, metric, group, chart_type=kind, agg=agg)
        elif kind == "Histogram":
            fig = polish(px.histogram(df, x=metric, nbins=35, color_discrete_sequence=COLORS))
            tab = df[[metric]].describe().T
        elif kind == "Box":
            if group in df.columns and df[group].nunique() <= 20:
                fig = polish(px.box(df, x=group, y=metric, points=False, color_discrete_sequence=COLORS))
            else:
                fig = polish(px.box(df, y=metric, points=False, color_discrete_sequence=COLORS))
            tab = df[[metric]].describe().T
        elif kind == "Scatter":
            second = st.selectbox("Second numeric axis", [n for n in num if n != metric] or [metric])
            sample = df.sample(min(10000, len(df)), random_state=42)
            fig = polish(px.scatter(sample, x=second, y=metric, opacity=.55, color_discrete_sequence=COLORS))
            tab = sample[[second, metric]].describe().T
            if len(df) > len(sample):
                st.caption("Scatter plots display a reproducible sample of up to 10,000 rows.")
        else:
            fig = correlation_heatmap(df)
            if fig is None:
                st.info("Correlation needs two or more numeric columns.")
                return
            tab = df.select_dtypes(include="number").corr(method="spearman")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("See values behind this chart"):
            result_table(tab)
            st.download_button("Export chart data", tab.to_csv(index=False).encode("utf-8"), "chart_data.csv", "text/csv")
        st.caption("Tip: move your mouse over charts to see exact values, and use the chart toolbar to export a PNG.")
    except Exception as exc:
        st.error(f"Could not draw this chart: {exc}")


def statistics_page(df: pd.DataFrame, mapping: Mapping) -> None:
    div_label("UNDER THE HOOD", "Statistical analysis", "Understand distributions, differences, missing values, outliers and relationships.")
    one, two, three = st.columns(3)
    with one:
        tiny_card("Missing cells", format_number(int(df.isna().sum().sum())), "Blank values across all columns")
    with two:
        tiny_card("Duplicate records", format_number(int(df.duplicated().sum())), "Exact duplicates; may be valid repeat records", "orange")
    with three:
        tiny_card("Numeric variables", str(len(df.select_dtypes(include="number").columns)), "Available for statistical checks", "violet")
    st.write("")
    st.markdown("#### Column quality")
    result_table(quality_profile(df))
    st.markdown("#### Descriptive statistics")
    stattab = summary_stats(df)
    if stattab.empty:
        st.info("Add a numeric field to see averages, quartiles and spread.")
        return
    result_table(stattab.round(3))
    st.caption("Mean = average; median = middle value; standard deviation = how spread out the values are.")
    st.markdown("#### Spot unusual values")
    nums = list(df.select_dtypes(include="number").columns)
    variable = st.selectbox("Numeric column", nums, index=nums.index(mapping.measure) if mapping.measure in nums else 0)
    outliers, lower, upper = outlier_profile(df, variable)
    st.write(f"IQR rule flagged **{len(outliers):,}** records outside the approximate range **{lower:,.2f} to {upper:,.2f}**. These are worth checking, not automatically errors.")
    if len(outliers):
        with st.expander("Inspect flagged rows"):
            result_table(outliers.head(200))
    st.markdown("#### Check statistical differences")
    cat = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 30]
    if cat:
        test_options = ["Compare two group averages (Welch t-test)"]
        if len(cat) >= 2:
            test_options.append("Relationship between categories (Chi-square)")
        test = st.radio("Analysis", test_options, horizontal=True)
        try:
            if test.startswith("Compare"):
                col = st.selectbox("Group variable", cat, index=cat.index(mapping.dimension) if mapping.dimension in cat else 0)
                options = sorted(df[col].dropna().astype(str).unique())
                ca, cb, cv = st.columns(3)
                with ca:
                    first = st.selectbox("Group A", options, index=0)
                with cb:
                    second = st.selectbox("Group B", options, index=min(1, len(options)-1))
                with cv:
                    number = st.selectbox("Number to compare", nums, index=nums.index(mapping.measure) if mapping.measure in nums else 0, key="ttest_num")
                if st.button("Run group comparison"):
                    if first == second:
                        st.warning("Pick two different groups.")
                    else:
                        res = welch_test(df, col, number, first, second)
                        st.metric("Difference in average (A − B)", f"{res['mean_difference']:,.2f}")
                        st.write(f"p-value: **{res['p_value']:.4g}** · approximate 95% confidence interval: **[{res['ci_low']:,.2f}, {res['ci_high']:,.2f}]**")
                        st.caption("A low p-value suggests a difference under test assumptions; it does not establish a causal effect. Test is row-level and can be misleading for repeated customers or dependent observations.")
            else:
                left, right = st.columns(2)
                with left:
                    first = st.selectbox("First category", cat)
                with right:
                    second = st.selectbox("Second category", [c for c in cat if c != first])
                if st.button("Run relationship test"):
                    res, table = chi_square_test(df, first, second)
                    st.write(f"Chi-square p-value **{res['p_value']:.4g}** · Cramér's V **{res['cramers_v']:.3f}**")
                    if res["low_expected"]:
                        st.warning(f"{res['low_expected']} expected frequencies fall below 5; consider merging rare categories before trusting the approximation.")
                    result_table(table)
                    st.caption("Association between categories does not establish causation.")
        except (ValueError, TypeError) as exc:
            st.warning(str(exc))
    else:
        st.info("A categorical column with 2–30 unique groups is needed for statistical tests.")
    st.markdown("#### Numeric relationships")
    fig = correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation reflects co-movement, not cause-and-effect.")


def forecasts(df: pd.DataFrame, mapping: Mapping) -> None:
    div_label("FORECAST, WITH CONTEXT", "Trends & anomaly alerts", "Incomplete months are excluded from model training. Forecasts are tested against historical periods.")
    if not mapping.date or not mapping.measure:
        st.info("Set a date and a numeric measure in the sidebar to enable forecasting.")
        return
    left, right = st.columns(2)
    with left:
        freq = st.selectbox("Time grouping", ["Weekly", "Monthly"], index=1)
    with right:
        horizon = st.slider("Periods ahead", 1, 12, 4)
    fcode = "W" if freq == "Weekly" else "MS"
    history = aggregate_series(df, mapping.date, mapping.measure, freq=fcode)
    series = aggregate_series(df, mapping.date, mapping.measure, freq=fcode, complete_only=True)
    if history.empty:
        st.warning("No valid time series values are available.")
        return
    excluded = history[~history["period"].isin(series["period"])]
    if not excluded.empty:
        names = ", ".join(pd.to_datetime(excluded["period"]).dt.strftime("%b %Y" if fcode == "MS" else "%d %b %Y").tolist()[:4])
        st.warning(f"Partial boundary periods excluded from forecast training: {names}. You can still inspect those values in the main dashboard.")
    if len(series) < 7:
        st.info(f"Only {len(series)} complete periods found; at least 7 are needed for a defensible backtest. Historical data remains available below.")
        if len(history):
            result_table(history.tail(30))
        return
    marked = flag_anomalies(series)
    forecast = None
    dates = None
    try:
        forecast, details = forecast_with_backtest(marked, horizon=horizon, freq=fcode)
        dates = pd.date_range(marked["period"].iloc[-1], periods=horizon+1, freq=fcode)[1:]
        a, b, c = st.columns(3)
        with a:
            tiny_card("Selected method", details["method"], "Best rolling one-period backtest", "red")
        with b:
            tiny_card("Backtest MAE", format_number(details["mae"]), f"Over {details['backtest_periods']} held-out periods", "teal")
        with c:
            tiny_card("Backtest sMAPE", f"{details['smape_pct']:.1f}%", "Scale-aware historical error", "violet")
        st.caption("Compared methods by rolling holdout MAE: " + ", ".join(f"{method} = {score:,.0f}" for method, score in details["candidate_mae"].items()))
        st.caption(details["interval_note"] + " Backtest results on the included sample are not proof of performance after deployment.")
    except ValueError as exc:
        st.info(str(exc))
    fig = anomaly_chart(marked, forecast, dates)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Unusual completed periods detected:** {int(marked['anomaly'].sum())} (median-based diagnostic, not automatic proof of an issue).")
    if forecast is not None:
        output = forecast.copy()
        output.insert(0, "period", dates)
        with st.expander("Inspect forecast values and uncertainty"):
            result_table(output.round(2))
            st.download_button("Download forecast CSV", output.to_csv(index=False).encode("utf-8"), "nexora_forecast.csv", "text/csv")
    st.caption("Forecasts exclude partial periods; validate business context, promotions, seasonality and data completeness before relying on estimates.")


def sql_lab(df: pd.DataFrame, mapping: Mapping) -> None:
    div_label("FOR POWER USERS", "SQL workbench", "Every query runs only against your uploaded table named `data`, with read-only validation.")
    with st.expander("Table schema", expanded=False):
        schema = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
        result_table(schema)
    templates = {"Count rows": "SELECT COUNT(*) AS total_rows FROM data",
                 "Preview data": "SELECT * FROM data LIMIT 20"}
    if mapping.dimension and mapping.measure:
        templates["Compare groups"] = f'SELECT "{mapping.dimension.replace(chr(34), chr(34)*2)}" AS segment, SUM("{mapping.measure.replace(chr(34), chr(34)*2)}") AS total FROM data GROUP BY 1 ORDER BY total DESC LIMIT 20'
    name = st.selectbox("Start from a safe example", list(templates))
    st.text_area("SQL query", value=templates[name], height=180, key=f"sql_editor_{name}")
    sql = st.session_state[f"sql_editor_{name}"]
    if st.button("Run SQL safely", type="primary"):
        try:
            output, truncated = execute_select(df, sql)
            st.session_state["sql_result"] = (output, truncated, sql)
        except Exception as exc:
            st.session_state.pop("sql_result", None)
            st.error(str(exc))
    saved = st.session_state.get("sql_result")
    if saved and saved[2] == sql:
        output, truncated, _ = saved
        result_table(output)
        if truncated:
            st.warning("Result truncated to 1,000 rows.")
        st.download_button("Download query output", output.to_csv(index=False).encode(), "sql_result.csv", "text/csv")
    st.caption("Only SELECT queries are accepted. File, network and database-changing operations are blocked. Do not treat the sandbox as a substitute for enterprise-grade data isolation.")


def reports(df: pd.DataFrame, mapping: Mapping, source: str) -> None:
    div_label("TAKE THE ANSWERS WITH YOU", "Reports & exports", "Branded business reports with the exact region and analysis mode you choose.")
    values = sorted(df[mapping.dimension].dropna().astype(str).unique()) if mapping.dimension else []
    defaults = ["Auto-select lowest total"] + values
    saved = st.session_state.get("last_diagnosed_group")
    default_index = defaults.index(saved) if saved in defaults else 0
    left, right = st.columns([1.4, 1])
    with left:
        selected = st.selectbox("Report focus", defaults, index=default_index,
                                help="Choose the segment used in the focused analysis. Previous investigations default here.")
    with right:
        kind = st.selectbox("Analysis in report", ["Peer comparison · whole dataset", "Period change · equal-length windows"],
                            index=1 if st.session_state.get("last_diagnosis_mode") == "trend" else 0)
    focus = selected if selected != defaults[0] else None
    mode = "trend" if kind.startswith("Period") else "level"
    days = st.selectbox("Time-window length", [30, 7, 60, 90], index=0) if mode == "trend" else 30
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown("#### Executive report · PDF + HTML")
        st.write("Professional branding, KPIs, peer comparisons, category evidence, charts and actionable recommendations.")
        try:
            html_report = build_html_report(df, mapping, source, focus=focus, mode=mode, days=days)
            st.download_button("Download branded HTML report", html_report.encode("utf-8"),
                               "nexora_insights_executive.html", "text/html", use_container_width=True)
            pdf_report = build_pdf_report(df, mapping, source, focus=focus, mode=mode, days=days)
            st.download_button("Download executive PDF", pdf_report, "nexora_insights_executive.pdf", "application/pdf",
                               use_container_width=True, type="primary")
        except Exception as exc:
            st.error(f"Report could not be generated: {exc}")
    with b:
        st.markdown("#### Filtered dataset · CSV")
        st.write("Export the current filtered view for Excel, colleagues or further checks.")
        st.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "nexora_filtered_data.csv", "text/csv", use_container_width=True)
        st.caption("Reports use the same data view and column mappings shown in the sidebar.")
    st.divider()
    st.markdown("#### Before sharing with clients")
    st.markdown("Do not upload confidential personal/customer data to a publicly accessible instance without authentication, data-isolation controls and a privacy review. PDF, HTML and CSV downloads contain business data and may be readable by anyone you share them with.")


def main() -> None:
    df, mapping, label, messages = sidebar_data()
    if df is None or mapping is None:
        st.title("NEXORA INSIGHTS")
        st.info("Upload a valid CSV to continue, or refresh to explore the included demo.")
        return
    st.markdown('<div class="top-line"><span class="status-dot"></span> NEXORA ANALYTICS WORKSPACE <span class="top-sign">CREATED BY HITESH SHARMA</span></div>', unsafe_allow_html=True)
    if messages:
        with st.expander("Data import notes"):
            for note in messages:
                st.caption("• " + note)
    nav = ["Overview", "Ask your data", "Explore charts", "Why did it change?", "Statistics", "Forecast", "Reports", "SQL lab"]
    st.sidebar.markdown("<div class='sidebar-step'>04 · OPEN A WORKSPACE</div>", unsafe_allow_html=True)
    st.sidebar.radio("Navigate", nav, key="nav", label_visibility="collapsed")
    page = st.session_state["nav"]
    if page == "Overview":
        render_hero(label, len(df), "SAMPLE DATA" in label)
        overview(df, mapping)
    elif page == "Ask your data": questions(df, mapping)
    elif page == "Explore charts": explorer(df, mapping)
    elif page == "Why did it change?": diagnostics_page(df, mapping)
    elif page == "Statistics": statistics_page(df, mapping)
    elif page == "Forecast": forecasts(df, mapping)
    elif page == "Reports": reports(df, mapping, label)
    else: sql_lab(df, mapping)
    st.markdown('<div class="footer">NEXORA INSIGHTS <span>Created by Hitesh Sharma · Evidence before assumptions.</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
