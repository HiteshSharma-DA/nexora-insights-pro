# NEXORA INSIGHTS — Test & release checklist

## Local test (before GitHub)

- [ ] Run `python -m pip install -r requirements.txt` and `python -m unittest discover -s tests -v`.
- [ ] Run `python -m streamlit run app.py`; check Overview shows the synthetic 5,301-row demo.
- [ ] Ask: **Why are North sales lower than other regions?** Confirm *whole-dataset peer comparison* with charts, SQL evidence and actions.
- [ ] Ask: **Why did North sales drop recently?** Confirm *equal-length period change* with SQL evidence.
- [ ] Go to Forecast. Confirm incomplete **Nov 2026** is flagged/excluded from training (for the included demo).
- [ ] Run a SQL Lab `SELECT COUNT(*) FROM data` and download CSV result.
- [ ] Run the Welch t-test in Statistics; explore the charts.
- [ ] Go to Reports and select **North**, **Peer comparison**. Open the downloaded HTML and PDF; confirm the selected group is North and bold text is formatted without `**` characters.
- [ ] Upload a different, non-sensitive CSV (e.g., one with `revenue`, `region`, `order_date` columns). Verify remapped columns and dashboard totals.
- [ ] Ask why its lowest region has low revenue. Verify it does **not** treat an empty recent date window as a confirmed -100% decline.
- [ ] Confirm loading errors, missing columns and insufficient forecast history are explained clearly.

## Public testing

- [ ] GitHub repo has `app.py`, `requirements.txt`, `core`, `assets`, `demo`, `tests`, and `.streamlit/config.toml`.
- [ ] No `.env`, `.venv`, API keys, personal/customer data or confidential report files are committed.
- [ ] Streamlit Cloud builds successfully and the public HTTPS app loads on desktop and mobile.
- [ ] Check PDF export and read-only SQL execution on the deployment.
- [ ] Invite testers to use **synthetic/non-sensitive files only** until access controls and privacy protections are in place.
- [ ] If enabling the optional AI key: control spending, rate limits, abuse and access permissions first.
- [ ] Collect feedback about confusing labels, unsupported question types, slow queries and inaccessible charts.

**Note:** NEXORA is a showcase/testing MVP. Multi-tenant security, account management, billing, retention policies and full production monitoring are still required for a commercial SaaS launch.
