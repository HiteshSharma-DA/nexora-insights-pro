"""Run after pip install -r requirements.txt: python -m unittest discover -s tests -v"""
from pathlib import Path
import unittest
import pandas as pd

from core.ingestion import load_csv, prepare_dates
from core.metadata import guess_mapping
from core.statistics import summary_stats, welch_test
from core.forecasting import aggregate_series, forecast_linear_trend
from core.sql_engine import execute_select, validate_select
from core.diagnostics import diagnose_segment
from core.agent import local_query

DATA_PATH = Path(__file__).resolve().parents[1] / 'demo' / 'retail_sales_demo.csv'


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, warnings = load_csv(DATA_PATH.read_bytes())
        cls.mapping = guess_mapping(cls.df)
        cls.df, bad = prepare_dates(cls.df, cls.mapping.date)
        assert bad == 0

    def test_auto_mapping(self):
        self.assertEqual(self.mapping.measure, 'Sales')
        self.assertEqual(self.mapping.dimension, 'Region')
        self.assertEqual(self.mapping.date, 'Order Date')
        self.assertEqual(self.mapping.category, 'Category')
        self.assertEqual(self.mapping.order_id, 'Order ID')

    def test_descriptive_statistics(self):
        out = summary_stats(self.df)
        self.assertIn('Sales', out['column'].tolist())
        res = welch_test(self.df, 'Region', 'Sales', 'North', 'South')
        self.assertGreater(res['n_a'], 50)
        self.assertGreaterEqual(res['p_value'], 0)

    def test_forecast(self):
        values = aggregate_series(self.df, 'Order Date', 'Sales', 'W')
        preds, metadata = forecast_linear_trend(values, 4)
        self.assertEqual(len(preds), 4)
        self.assertEqual(metadata['method'][:12], 'Linear trend')

    def test_sql_runs_in_memory(self):
        result, truncated = execute_select(self.df, 'SELECT COUNT(*) AS count_rows FROM data')
        self.assertEqual(int(result.iloc[0]['count_rows']), len(self.df))
        self.assertFalse(truncated)

    def test_sql_read_only(self):
        for sql in [
            "DROP TABLE data",
            "DELETE FROM data",
            "SELECT * FROM read_csv('/etc/passwd')",
            "SELECT 1; SELECT 2",
            "COPY data TO '/tmp/test.csv'",
        ]:
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                validate_select(sql)

    def test_sql_templates(self):
        sql, label = local_query('Show top 5 regions by sales', self.df, self.mapping)
        result, _ = execute_select(self.df, sql)
        self.assertEqual(len(result), 4)
        self.assertIn('total_measure', result.columns)

    def test_why_diagnosis_has_sql_evidence(self):
        report = diagnose_segment(self.df, self.mapping, 'North', days=30)
        self.assertTrue(report.explanation)
        self.assertTrue(report.actions)
        self.assertGreaterEqual(len(report.queries), 2)
        self.assertIn('North', report.headline)


if __name__ == '__main__':
    unittest.main()


class UpgradeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from core.ingestion import load_csv, prepare_dates
        from core.metadata import guess_mapping
        cls.df, _ = load_csv(DATA_PATH.read_bytes())
        cls.mapping = guess_mapping(cls.df)
        cls.df, _ = prepare_dates(cls.df, cls.mapping.date)

    def test_natural_language_routes_low_vs_declining(self):
        from core.diagnostics import question_intent
        self.assertEqual(question_intent('Why are North sales lower than other regions?'), 'level')
        self.assertEqual(question_intent('Why is West revenue lowest?'), 'level')
        self.assertEqual(question_intent('Why did North revenue drop last month?'), 'trend')
        self.assertEqual(question_intent('Why has West sales decreased?'), 'trend')

    def test_level_investigation_compares_all_regions(self):
        from core.diagnostics import diagnose_segment
        report = diagnose_segment(self.df, self.mapping, 'North', mode='level')
        self.assertEqual(report.mode, 'level')
        self.assertEqual(len(report.summary), 4)
        self.assertGreaterEqual(len(report.queries), 2)
        self.assertTrue(any('Full available dataset' in x for x in report.explanation))
        self.assertFalse(any('Comparable windows' in x for x in report.explanation))

    def test_partial_month_is_not_used_in_training(self):
        from core.forecasting import aggregate_series, forecast_with_backtest
        full = aggregate_series(self.df, 'Order Date', 'Sales', 'MS')
        complete = aggregate_series(self.df, 'Order Date', 'Sales', 'MS', complete_only=True)
        self.assertEqual(len(full), len(complete) + 1)
        self.assertEqual(complete['period'].max().strftime('%Y-%m'), '2026-10')
        result, quality = forecast_with_backtest(complete, horizon=4, freq='MS')
        self.assertEqual(len(result), 4)
        self.assertGreater(quality['backtest_periods'], 0)
        self.assertGreaterEqual(quality['mae'], 0)

    def test_branded_report_focus_and_pdf(self):
        from core.reports import build_html_report, build_pdf_report
        report = build_html_report(self.df, self.mapping, 'Retail demo', focus='North', mode='level')
        self.assertIn('NEXORA', report)
        self.assertIn('Focused investigation: North', report)
        self.assertIn('Technical appendix', report)
        self.assertNotIn('**North', report)
        pdf = build_pdf_report(self.df, self.mapping, 'Retail demo', focus='North', mode='level')
        self.assertTrue(pdf.startswith(b'%PDF-'))
        self.assertGreater(len(pdf), 1500)

    def test_sparse_period_does_not_claim_decline(self):
        from core.diagnostics import diagnose_segment
        sparse = self.df.loc[self.df['Region'].eq('North')].head(10).copy()
        # A date window with no rows in the last 30 days for North.
        other = self.df.loc[self.df['Region'].eq('West')].tail(15).copy()
        other['Order Date'] = pd.Timestamp('2026-11-15')
        sparse = __import__('pandas').concat([sparse, other], ignore_index=True)
        outcome = diagnose_segment(sparse, self.mapping, 'North', mode='trend', days=30)
        self.assertTrue(any('Insufficient period coverage' in x for x in outcome.explanation))
        self.assertFalse(any('-100.0% change' in x for x in outcome.explanation))
