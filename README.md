# NEXORA INSIGHTS — Decision Intelligence

**Created by Hitesh Sharma** · premium charcoal + crimson Streamlit analytics application.

NEXORA transforms uploaded CSV data into a business dashboard, SQL-backed answers, whole-dataset group benchmarks, period-change diagnostics, statistical comparisons, interactive charts, forecasts with historical backtests, and professional **PDF + HTML** reports. The demo is synthetic; no OpenAI key is needed for the built-in dashboards, diagnostics or supported question templates.

## Quick start on Windows

1. Extract the ZIP and open the **`nexora_insights_pro`** folder (the folder that contains `app.py`).
2. **Easy mode:** double-click **`START_NEXORA_WINDOWS.bat`**. The script creates a separate `.venv`, installs requirements and launches Streamlit.
3. **VS Code mode:** open Terminal inside `nexora_insights_pro` and run:

   ```powershell
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. Open `http://localhost:8501` if it does not open automatically. Keep the terminal open while testing.
5. Start with the built-in demo, or upload your own `.csv`. Under **Adjust column meanings**, check that the selected number, group, date, category and order ID match your data.

Python **3.11, 3.12 or 3.13** is recommended, with network access for the initial package installation. The one-click script keeps dependencies separate from your other Python projects.

## What's included (original features retained)

| Page | What it does |
|---|---|
| Overview | KPI cards, time trends, region charts, automatic highlights and data preview |
| Ask your data | Simple questions, SQL evidence, downloads; optional OpenAI-generated guarded SQL |
| Explore charts | Bar, line, pie, treemap, histogram, box, scatter and correlation visuals |
| Why did it change? | **Choose:** whole-dataset peer comparison *or* equal-length period-change analysis; category gap analysis and actions |
| Statistics | Completeness, duplicates, descriptions, IQR outliers, Welch's t-test, chi-square, Spearman correlations |
| Forecast | Conservative exclusion of partial boundary periods, rolling backtests, MAE/RMSE/sMAPE, trend/naive comparison, anomalies |
| Reports | Focus on the selected segment, branded **PDF**, responsive HTML with charts and optional SQL appendix, filtered CSV |
| SQL lab | Manual read-only DuckDB queries, table schema, examples, CSV download |

### Correct question interpretation

**"Why is West revenue lower than other regions?"** → compares the **entire current filtered dataset** using segment total, number of records, average revenue per row, product/category gaps and SQL. It does **not** invent a falling trend.

**"Why did West revenue drop last month?"** → compares equal-length recent periods, with explicit missing/zero-window coverage warnings. A zero-row window is **not** automatically proof of a 100% sales collapse.

If a question cannot be supported by available columns, use the optional AI SQL generator or SQL Lab. Built-in Q&A is intentionally scoped—it is not unrestricted general-purpose AI.

### Forecast limitations

- The app **conservatively** excludes the earliest/latest incomplete time buckets in forecasting. If your dataset is backfilled with days that contain zero transactions, a boundary can be falsely labelled incomplete—confirm coverage in the source data.
- Backtests compare **last-period** and **linear-trend** baselines; seasonal naive is tried only with sufficient historical periods. The uncertainty region is **indicative**, not a calibrated 95% confidence interval.
- Forecasts are not production-validated and should not be sold as guaranteed sales predictions.

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository and upload the files **inside `nexora_insights_pro/`** so `app.py` and `requirements.txt` are at the repository root. GitHub upload should include `.streamlit/config.toml`, `assets/`, `core/`, `demo/` and `tests/`. Do **not** upload `.venv/`, secret keys, private CSV files or customer reports.
2. On [share.streamlit.io](https://share.streamlit.io) choose **Create app**, select the repo and branch, and set **main file path = `app.py`**.
3. Deploy and test every page with the synthetic demo first. Then use a non-sensitive sample CSV with different column names.
4. For optional AI, in the app's **Advanced settings / Secrets**, put:

   ```toml
   OPENAI_API_KEY = "your-secret-api-key"
   ```

   Never commit the key to source code. AI calls cost money. Public app traffic can incur charges—add authentication, budgets, quotas and abuse protection *before* enabling public AI access.
5. Use the deployed HTTPS URL for testers. `localhost:8501` is only accessible on your own computer.

**A new ZIP does not automatically update your older Streamlit link**. Replace the files in its linked GitHub repository, or deploy this as a separate app, then change its source link.

## Testing

From the project folder after installation:

```powershell
python -m unittest discover -s tests -v
```

Original automated tests are retained and expanded to check question intent, peer comparisons, partial-month handling, PDF/HTML reports and zero-row periods. Before public sharing, also perform a browser-based pass on the deployed Streamlit app.

## Data and security boundaries

- CSV limit: **50 MB**, **350,000 rows**, **150 columns**; data is stored in the Streamlit app session and analysed in memory.
- Built-in diagnostics use the user's uploaded rows; optional AI sends **question and schema (column names/types and role mapping)**, not raw dataset rows, to the AI service. Column names themselves may be sensitive.
- This is a **shareable testing MVP**, **not a certified multi-tenant SaaS**. Public Streamlit deployments need sign-in, per-user isolation, rate limits, retention/deletion policy, resource limits and privacy review before accepting confidential business/customer information.
- Statistical significance and correlation do not establish causality; low regional revenue can reflect market size, coverage, quantity and unit prices—not necessarily poor performance.
- Exported PDF/HTML/CSV files may expose the data they contain. Share intentionally.

## Project structure

```text
nexora_insights_pro/
├── app.py
├── requirements.txt
├── START_NEXORA_WINDOWS.bat
├── .streamlit/config.toml
├── assets/style.css
├── core/ (ingestion, metadata, SQL engine, agent, diagnostics, statistics, forecasting, charts, reports)
├── demo/retail_sales_demo.csv (synthetic demo)
└── tests/test_core.py
```

**Deployment check:** complete = local import/run + GitHub deployment + test user review. The downloaded package contains the code but does not deploy itself.
