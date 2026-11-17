"""Consistent plotting helpers with deliberately bounded categories/points."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = ["#D93651", "#252027", "#D47B82", "#8B516B", "#E5A55A", "#85838B"]


def polish(fig, title: str | None = None):
    fig.update_layout(
        title={"text": title or "", "font": {"size": 17}},
        template="plotly_white", colorway=COLORS,
        font={"family": "Inter, Arial, sans-serif", "color": "#2B2028"},
        margin=dict(l=12, r=15, t=65 if title else 25, b=30),
        legend_title_text="", height=390,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    )
    return fig


def by_dimension(df: pd.DataFrame, measure: str, dimension: str, chart_type: str = "Bar", agg: str = "Sum", top_n: int = 15):
    data = df[[measure, dimension]].dropna(subset=[measure, dimension]).copy()
    data[dimension] = data[dimension].astype(str)
    if agg == "Sum":
        tab = data.groupby(dimension, as_index=False, observed=True)[measure].sum()
    elif agg == "Mean":
        tab = data.groupby(dimension, as_index=False, observed=True)[measure].mean()
    else:
        tab = data.groupby(dimension, as_index=False, observed=True)[measure].count()
    tab = tab.sort_values(measure, ascending=False).head(top_n)
    if chart_type == "Pie":
        fig = px.pie(tab, names=dimension, values=measure, hole=.5, color_discrete_sequence=COLORS)
    elif chart_type == "Treemap":
        fig = px.treemap(tab, path=[dimension], values=measure, color_discrete_sequence=COLORS)
    elif chart_type == "Line":
        fig = px.line(tab, x=dimension, y=measure, markers=True)
    else:
        fig = px.bar(tab.sort_values(measure), x=measure, y=dimension, orientation="h", color_discrete_sequence=COLORS)
    return polish(fig), tab


def time_trend(df: pd.DataFrame, date: str, measure: str, freq: str = "M", dimension: str | None = None):
    data = df.dropna(subset=[date, measure]).copy()
    if data.empty:
        return None
    data["period"] = data[date].dt.to_period(freq).dt.to_timestamp()
    grouping = ["period"] + ([dimension] if dimension else [])
    grouped = data.groupby(grouping, dropna=False, observed=True)[measure].sum().reset_index()
    if dimension and grouped[dimension].nunique() > 12:
        top = data.groupby(dimension)[measure].sum().nlargest(12).index
        grouped = grouped[grouped[dimension].isin(top)]
    fig = px.line(grouped, x="period", y=measure, color=dimension, markers=True, color_discrete_sequence=COLORS)
    return polish(fig)


def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number").iloc[:, :25]
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(method="spearman")
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=[[0,"#50424C"],[0.5,"#F5F0F2"],[1,"#D93651"]], aspect="auto")
    fig.update_layout(height=max(410, len(corr.columns) * 30))
    return polish(fig, "Spearman rank correlations")


def anomaly_chart(series: pd.DataFrame, forecast: pd.DataFrame | None = None, next_dates=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["period"], y=series["value"], mode="lines+markers", name="Actual", line=dict(color=COLORS[0])))
    anomalies = series[series["anomaly"]]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["period"], y=anomalies["value"], mode="markers", name="Anomaly", marker=dict(color="#DC2626", size=12, symbol="diamond")))
    if forecast is not None and next_dates is not None:
        fig.add_trace(go.Scatter(x=next_dates, y=forecast["prediction"], mode="lines+markers", name="Trend estimate", line=dict(dash="dash", color=COLORS[1])))
        fig.add_trace(go.Scatter(x=list(next_dates) + list(reversed(next_dates)), y=list(forecast["upper"]) + list(reversed(forecast["lower"])), fill="toself", fillcolor="rgba(217,54,81,.12)", line=dict(color="rgba(0,0,0,0)"), name="Indicative forecast range"))
    return polish(fig)
