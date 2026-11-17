"""Branded HTML and PDF executive reports, grounded in the selected data/view."""
from __future__ import annotations
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import re
import pandas as pd
from .metadata import Mapping
from .diagnostics import diagnose_segment

BRAND = "NEXORA INSIGHTS"
RED = "#E53E52"
DARK = "#14131A"


def _rich(value: object) -> str:
    """Escape dynamic content; display only the **bold** syntax from generated text."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escape(str(value)))


def _table(df: pd.DataFrame, rows: int = 20) -> str:
    return df.head(rows).to_html(index=False, classes="data-table", border=0, escape=True,
                                 float_format=lambda x: f"{x:,.2f}")


def _region_summary(df: pd.DataFrame, mapping: Mapping) -> pd.DataFrame:
    if not mapping.measure or not mapping.dimension:
        return pd.DataFrame()
    summary = (df.groupby(mapping.dimension, dropna=False, observed=True)[mapping.measure]
               .agg(["sum", "count", "mean"]).reset_index().sort_values("sum", ascending=False))
    summary.columns = [mapping.dimension, "Total", "Records", "Average per row"]
    return summary


def _svg_bars(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    data = summary.head(10)
    w, label_w, bar_w = 880, 150, 620
    h = len(data)*48 + 32
    peak = max(float(data["Total"].abs().max()), 1)
    parts = [f'<svg role="img" aria-label="Segment totals chart" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for i, (_, row) in enumerate(data.iterrows()):
        y = 18+i*48
        name = escape(str(row.iloc[0])[:22])
        val = float(row["Total"])
        bar_width = max(1, min(bar_w, abs(val)/peak*bar_w))
        parts.extend([
            f'<text x="4" y="{y+20}" fill="#343039" font-size="15" font-family="Arial">{name}</text>',
            f'<rect x="{label_w}" y="{y+5}" rx="5" width="{bar_w}" height="24" fill="#F2E9EA"/>',
            f'<rect x="{label_w}" y="{y+5}" rx="5" width="{bar_width:.1f}" height="24" fill="{RED}"/>',
            f'<text x="{label_w+bar_w+8}" y="{y+22}" fill="#17141C" font-size="14" font-family="Arial">{val:,.0f}</text>',
        ])
    parts.append("</svg>")
    return "".join(parts)


def _mini_trend(df: pd.DataFrame, mapping: Mapping) -> str:
    if not mapping.date or not mapping.measure or not pd.api.types.is_datetime64_any_dtype(df[mapping.date]):
        return ""
    values = df.dropna(subset=[mapping.date, mapping.measure]).copy()
    if values.empty:
        return ""
    trend = values.groupby(values[mapping.date].dt.to_period("M"), observed=True)[mapping.measure].sum()
    if len(trend) < 2:
        return ""
    series = trend.tail(24)
    a, b = float(series.min()), float(series.max())
    span = b-a if b>a else 1
    pts = [f"{35+i*780/max(1,len(series)-1):.1f},{195-(float(v)-a)/span*140:.1f}" for i,v in enumerate(series)]
    return ('<div class="chart-wrap"><h3>Monthly totals (includes recorded partial months)</h3>'
            '<svg role="img" aria-label="Historical monthly totals" viewBox="0 0 860 230" xmlns="http://www.w3.org/2000/svg">'
            '<line x1="30" x2="830" y1="200" y2="200" stroke="#C6C0C4"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RED}" stroke-width="5" stroke-linejoin="round"/>'
            '</svg></div>')


def _pretty_drivers(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"driver": "Category / product", "target_total": "Selected total",
                              "target_rows": "Rows", "peer_mean_total": "Peer average",
                              "difference": "Difference", "peer_groups_present": "Peer groups",
                              "current_value": "Current", "previous_value": "Previous", "delta": "Change"})


def _report_parts(df: pd.DataFrame, mapping: Mapping, label: str,
                  focus: str | None, mode: str, days: int):
    summary = _region_summary(df, mapping)
    if focus is None and not summary.empty:
        focus = str(summary.iloc[-1, 0])
    diag = None
    if focus and mapping.dimension and mapping.measure:
        diag = diagnose_segment(df, mapping, focus, mode=mode, days=days)
    return summary, focus, diag


def build_html_report(df: pd.DataFrame, mapping: Mapping, label: str,
                      focus: str | None = None, mode: str = "level", days: int = 30) -> str:
    summary, focus, diag = _report_parts(df, mapping, label, focus, mode, days)
    stamp = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")
    metric = mapping.measure
    total = float(pd.to_numeric(df[metric], errors="coerce").sum()) if metric else None
    header = f'''<section class="cover"><div class="mast">NEXORA <b>INSIGHTS</b><span>DECISION INTELLIGENCE PLATFORM</span></div>
      <div class="by">CREATED BY HITESH SHARMA</div>
      <div class="eyebrow">EXECUTIVE BUSINESS REVIEW</div><h1>From numbers<br><em>to decisions.</em></h1>
      <p>A focused, evidence-led review of your uploaded data. Built for real business decisions.</p></section>
      <div class="meta"><span><strong>DATA SOURCE</strong> {escape(label)}</span><span><strong>GENERATED</strong> {stamp}</span></div>'''
    blocks = [header,
              '<div class="stats">' + ''.join(
                  f'<div><small>{name}</small><strong>{value}</strong></div>' for name,value in [
                      ("DATA ROWS", f"{len(df):,}"), ("COLUMNS", f"{len(df.columns):,}"),
                      ("MISSING CELLS", f"{int(df.isna().sum().sum()):,}"),
                      ("DUPLICATE ROWS", f"{int(df.duplicated().sum()):,}")]) + '</div>']
    if metric:
        blocks.append(f'<h2>Business overview</h2><p>The total recorded <b>{escape(metric)}</b> is <b>{total:,.2f}</b>. Null measurements are excluded from this sum.</p>')
    if not summary.empty:
        blocks.append(f'<h2>Performance by {escape(mapping.dimension)}</h2><p class="muted">Whole filtered dataset, not a selected time window. Rows are not necessarily unique orders.</p>')
        blocks.append('<div class="chart-wrap">'+_svg_bars(summary)+'</div>')
        blocks.append(_table(summary))
        blocks.append(_mini_trend(df, mapping))
    if diag:
        blocks.append(f'<h2>Focused investigation: {escape(focus)}</h2><div class="panel"><span class="pill">{("PERIOD CHANGE" if diag.mode == "trend" else "REGIONAL BENCHMARK")}</span>')
        blocks.extend(f'<p>{_rich(statement)}</p>' for statement in diag.explanation)
        if not diag.drivers.empty:
            blocks.append(f'<h3>{escape(mapping.category or "Category")} analysis</h3>'+_table(_pretty_drivers(diag.drivers), 12))
        blocks.append('</div><h3>Recommended next actions</h3><ol>')
        blocks.extend(f'<li>{_rich(action)}</li>' for action in diag.actions)
        blocks.append(f'</ol><p class="muted">{escape(diag.caveat)}</p>')
        blocks.append('<details class="sql-details"><summary>Technical appendix · executed SQL and evidence</summary>')
        for title, sql, evidence in diag.queries:
            blocks.append(f'<h4>{escape(title)}</h4><pre>{escape(sql)}</pre>' + _table(evidence, 12))
        blocks.append('</details>')
    quality = pd.DataFrame({"Column": df.columns, "Nulls": df.isna().sum().values,
                            "Null %": (df.isna().mean()*100).round(2).values})
    blocks.append('<h2>Data quality</h2>' + _table(quality, 150))
    blocks.append('<footer>NEXORA INSIGHTS <span>Created by Hitesh Sharma · Evidence before assumptions</span></footer>')
    css = '''<style>
      *{box-sizing:border-box}body{font-family:Arial,Helvetica,sans-serif;color:#242027;max-width:1150px;margin:0 auto;padding:32px;line-height:1.65;background:#faf8f9}
      .cover{background:linear-gradient(125deg,#151219,#24131c 67%,#801c33);color:white;padding:50px 55px;border-radius:23px;min-height:320px}
      .mast{font-size:21px;letter-spacing:2px;font-weight:900}.mast b{color:#ff506b}.mast span{font-size:9px;letter-spacing:3px;margin-left:20px;color:#f0b5bd}
      .by{font-size:10px;letter-spacing:2px;margin-top:10px;color:#d8c8cb}.eyebrow{font-size:11px;font-weight:800;letter-spacing:2px;color:#ff6379;margin-top:46px}
      h1{font-size:47px;line-height:1.1;margin:10px 0 17px;letter-spacing:-1.5px}h1 em{font-style:normal;color:#ff6278}.cover p{max-width:600px;color:#e4d9dd}
      .meta{display:flex;justify-content:space-between;gap:16px;padding:18px 2px;font-size:12px;color:#70666a}.meta strong{font-size:10px;letter-spacing:1px;margin-right:10px;color:#b83249}
      .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 36px}.stats div{background:white;border:1px solid #f0e6e8;border-radius:13px;padding:18px}
      .stats small{display:block;font-size:10px;color:#9a6974;letter-spacing:1px;font-weight:bold}.stats strong{display:block;font-size:27px;color:#201a20;margin-top:6px}
      h2{font-size:23px;margin:34px 0 14px;border-bottom:2px solid #e9d8dd;padding-bottom:10px;color:#17141b}h3{font-size:17px;margin-top:23px}
      .chart-wrap{background:white;padding:14px;border:1px solid #eee2e6;border-radius:16px;margin:14px 0;overflow:hidden}.chart-wrap svg{width:100%;height:auto}
      .data-table{border-collapse:collapse;width:100%;font-size:12px;background:white}.data-table th{background:#f4e9eb;color:#532230;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
      .data-table th,.data-table td{padding:11px;border-bottom:1px solid #eee3e7;text-align:left}.panel{padding:21px 24px;border:1px solid #e9d5dc;border-left:5px solid #e53e52;background:#fff;border-radius:8px}
      .sql-details{margin-top:22px;padding:16px 20px;background:white;border:1px solid #efd9df;border-radius:12px}.sql-details summary{cursor:pointer;color:#9e2745;font-weight:700}.sql-details pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#faf5f6;padding:10px;border-radius:6px}.pill{font-size:10px;letter-spacing:1px;color:#b83249;font-weight:bold}.muted{font-size:12px;color:#756a71}li{padding:4px 0}footer{margin-top:50px;border-top:1px solid #e3d8db;padding:20px 0;color:#a02d43;font-size:11px;font-weight:bold;display:flex;justify-content:space-between}footer span{color:#83757a;font-weight:normal}
      @media(max-width:650px){body{padding:10px}.cover{padding:32px 23px}.cover h1{font-size:36px}.stats{grid-template-columns:repeat(2,1fr)}.meta{display:block}footer{display:block}}
      @media print{body{background:white;max-width:none;padding:5mm}.cover{print-color-adjust:exact;-webkit-print-color-adjust:exact}.stats div,.data-table th{print-color-adjust:exact;-webkit-print-color-adjust:exact}.cover{break-inside:avoid}h2,h3{break-after:avoid}.panel,.chart-wrap{break-inside:avoid}}
      </style>'''
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NEXORA | Executive Insights Report</title>'+css+'</head><body>'+"\n".join(blocks)+'</body></html>'


def build_pdf_report(df: pd.DataFrame, mapping: Mapping, label: str,
                     focus: str | None = None, mode: str = "level", days: int = 30) -> bytes:
    """True downloadable PDF (ReportLab), no browser print step or API key."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.graphics.shapes import Drawing, Rect, String

    summary, focus, diag = _report_parts(df, mapping, label, focus, mode, days)
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=21*mm, bottomMargin=19*mm, title="NEXORA Insights Report", author="Hitesh Sharma")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="NxTitle", fontName="Helvetica-Bold", fontSize=26, leading=33, textColor=colors.HexColor(DARK), spaceAfter=12))
    styles.add(ParagraphStyle(name="NxSection", fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=colors.HexColor(DARK), spaceBefore=18, spaceAfter=9))
    styles.add(ParagraphStyle(name="NxBody", fontName="Helvetica", fontSize=9.4, leading=14.5, spaceAfter=9, textColor=colors.HexColor("#383039")))
    styles.add(ParagraphStyle(name="NxTiny", fontName="Helvetica", fontSize=8, leading=12, spaceAfter=8, textColor=colors.HexColor("#756971")))
    story = [Paragraph("NEXORA <font color='#E53E52'>INSIGHTS</font>", styles["NxSection"]),
             Paragraph("Business intelligence, built around your data", styles["NxTitle"]),
             Paragraph("Created by Hitesh Sharma", styles["NxBody"]),
             Paragraph("Source: " + escape(label) + " | " + datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"), styles["NxTiny"])]
    cells = [("DATA ROWS", f"{len(df):,}"), ("COLUMNS", str(len(df.columns))),
             ("MISSING CELLS", f"{int(df.isna().sum().sum()):,}"),
             ("DUPLICATE ROWS", f"{int(df.duplicated().sum()):,}")]
    kpis = Table([[Paragraph(f"<font size='7' color='#9D6877'>{k}</font><br/><font size='19'><b>{v}</b></font>", styles["NxBody"]) for k,v in cells]],
                 colWidths=[43.3*mm]*4)
    kpis.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#F9F1F3")),
                              ("BOX",(0,0),(-1,-1),.4,colors.HexColor("#EBD9DF")),
                              ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),13),
                              ("BOTTOMPADDING",(0,0),(-1,-1),12)]))
    story += [Spacer(1,8*mm), kpis]
    if mapping.measure:
        value = float(pd.to_numeric(df[mapping.measure], errors="coerce").sum())
        story += [Paragraph("Business overview", styles["NxSection"]),
                  Paragraph(f"Recorded {escape(mapping.measure)} total: <b>{value:,.2f}</b>. Missing measurements are excluded.", styles["NxBody"])]

    def as_table(data: pd.DataFrame, max_rows: int = 12):
        cols = list(data.columns)
        subset = data.head(max_rows)
        if not len(cols):
            return None
        width = 173*mm / len(cols)
        head = [Paragraph(escape(str(c)[:23]), styles["NxTiny"]) for c in cols]
        rows = [head]
        for _, row in subset.iterrows():
            rows.append([Paragraph(escape(f"{x:,.2f}" if isinstance(x, float) else str(x)[:36]), styles["NxTiny"]) for x in row.tolist()])
        table = Table(rows, colWidths=[width]*len(cols), repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), colors.HexColor("#F4E3E8")),
                                   ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#FCF9FA")]),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),6),
                                   ("LINEBELOW",(0,0),(-1,0),1,colors.HexColor(RED)),
                                   ("VALIGN",(0,0),(-1,-1),"TOP")]))
        return table

    if not summary.empty:
        story.append(Paragraph(f"Performance by {escape(mapping.dimension)}", styles["NxSection"]))
        d = Drawing(490, min(230, len(summary)*30+22))
        max_value = max(abs(float(summary["Total"].max())),1)
        height = d.height
        for index, (_, row) in enumerate(summary.head(7).iterrows()):
            y = height - 25 - index * 30
            d.add(String(3,y+4,str(row.iloc[0])[:17],fontSize=9,fillColor=colors.HexColor(DARK)))
            d.add(Rect(105,y,330,16,fillColor=colors.HexColor("#F3E6E9"),strokeColor=None))
            d.add(Rect(105,y,max(1,330*max(0,float(row["Total"]))/max_value),16,fillColor=colors.HexColor(RED),strokeColor=None))
            d.add(String(441,y+4,f"{float(row['Total'])/1000:,.0f}k",fontSize=8,fillColor=colors.HexColor(DARK)))
        story += [d, Spacer(1,4*mm),as_table(summary)]
    if diag:
        story.append(Paragraph("Focused investigation: "+escape(str(focus)), styles["NxSection"]))
        story.append(Paragraph("Period change" if diag.mode == "trend" else "Whole-dataset peer comparison", styles["NxTiny"]))
        for statement in diag.explanation:
            story.append(Paragraph(_rich(statement),styles["NxBody"]))
        if not diag.drivers.empty:
            story += [Paragraph("Category evidence", styles["NxSection"]),as_table(_pretty_drivers(diag.drivers).head(8))]
        story.append(Paragraph("Recommended actions", styles["NxSection"]))
        for idx, action in enumerate(diag.actions,1):
            story.append(Paragraph(str(idx)+". "+_rich(action),styles["NxBody"]))
        story.append(Paragraph(escape(diag.caveat), styles["NxTiny"]))
    quality = pd.DataFrame({"Column":df.columns,"Nulls":df.isna().sum().values,
                            "Null %":(df.isna().mean()*100).round(2).values})
    story += [Paragraph("Column quality",styles["NxSection"]),as_table(quality,30)]
    story = [p for p in story if p is not None]

    def footer(canvas, doc_obj):
        canvas.saveState()
        page_w, _ = A4
        canvas.setStrokeColor(colors.HexColor("#E4CDD3"))
        canvas.line(18*mm,15*mm,page_w-18*mm,15*mm)
        canvas.setFont("Helvetica",8)
        canvas.setFillColor(colors.HexColor("#AA304B"))
        canvas.drawString(18*mm,11*mm,"NEXORA INSIGHTS  |  Hitesh Sharma")
        canvas.drawRightString(page_w-18*mm,11*mm,f"Page {doc_obj.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
