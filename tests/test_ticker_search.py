"""Tests for GET /api/tickers/search autocomplete endpoint."""

import unittest
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app


class TestTickerSearchEndpoint(unittest.TestCase):

    def setUp(self):
        flask_app.app.config['TESTING'] = True
        self.client = flask_app.app.test_client()

    def _mock_search(self, query, limit=8):
        db = {
            'AAPL': 'Apple Inc.',
            'AAMT': 'AAMT CORP',
            'AAP':  'Advance Auto Parts',
            'GOOGL': 'Alphabet Inc.',
        }
        q = query.strip().upper()
        results = sorted(t for t in db if t.startswith(q))[:limit]
        return [{'ticker': t, 'name': db[t]} for t in results]

    def test_returns_results_for_valid_prefix(self):
        with patch('app._search_tickers', side_effect=self._mock_search):
            resp = self.client.get('/api/tickers/search?q=AA')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('results', data)
        tickers = [r['ticker'] for r in data['results']]
        self.assertIn('AAPL', tickers)
        self.assertIn('AAP', tickers)

    def test_returns_empty_for_blank_query(self):
        resp = self.client.get('/api/tickers/search?q=')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['results'], [])

    def test_respects_limit_parameter(self):
        with patch('app._search_tickers', side_effect=self._mock_search):
            resp = self.client.get('/api/tickers/search?q=A&limit=2')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertLessEqual(len(data['results']), 2)

    def test_result_items_have_ticker_and_name(self):
        with patch('app._search_tickers', side_effect=self._mock_search):
            resp = self.client.get('/api/tickers/search?q=AAPL')
        self.assertEqual(resp.status_code, 200)
        results = resp.get_json()['results']
        self.assertTrue(len(results) > 0)
        first = results[0]
        self.assertIn('ticker', first)
        self.assertIn('name', first)
        self.assertEqual(first['ticker'], 'AAPL')
        self.assertEqual(first['name'], 'Apple Inc.')


if __name__ == '__main__':
    unittest.main()
