"""
Edge-case tests for special market symbol news headline resolution.

Covers:
- SPECIAL_SYMBOL_TERMS completeness and structure
- _get_search_terms() priority and output for special symbols
- NewsAPI / Webz.io query building avoids raw special symbol
- Google News query list uses human names for special symbols
- Simulation fallback uses display_name and name-based URLs
- Standard tickers are unaffected (regression)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_mvp import SPECIAL_SYMBOL_TERMS, _get_search_terms


# ---------------------------------------------------------------------------
# 1. SPECIAL_SYMBOL_TERMS structure
# ---------------------------------------------------------------------------

class TestSpecialSymbolTermsStructure(unittest.TestCase):

    REQUIRED_KEYS = {"display_name", "names", "sector"}
    KNOWN_SYMBOLS = [
        "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^VIX",
        "CL=F", "GC=F", "SI=F", "HG=F", "NG=F",
        "^TNX", "^TYX", "DX-Y.NYB", "^FTSE", "^N225", "^HSI",
    ]

    def test_all_known_symbols_present(self):
        for sym in self.KNOWN_SYMBOLS:
            self.assertIn(sym, SPECIAL_SYMBOL_TERMS, f"Missing: {sym}")

    def test_each_entry_has_required_keys(self):
        for sym, entry in SPECIAL_SYMBOL_TERMS.items():
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, entry, f"{sym} missing key '{key}'")

    def test_display_name_is_non_empty_string(self):
        for sym, entry in SPECIAL_SYMBOL_TERMS.items():
            self.assertIsInstance(entry["display_name"], str)
            self.assertTrue(entry["display_name"].strip(), f"{sym} display_name is blank")

    def test_names_list_has_at_least_two_entries(self):
        for sym, entry in SPECIAL_SYMBOL_TERMS.items():
            self.assertGreaterEqual(
                len(entry["names"]), 2,
                f"{sym} should have >= 2 name synonyms for robust query building"
            )

    def test_names_contain_no_raw_special_symbols(self):
        """News API names must never be the raw ticker (^, =F, etc.)."""
        bad_patterns = ["^", "=F", "=X"]
        for sym, entry in SPECIAL_SYMBOL_TERMS.items():
            for name in entry["names"]:
                for pat in bad_patterns:
                    self.assertNotIn(
                        pat, name,
                        f"{sym} name '{name}' contains raw special symbol pattern '{pat}'"
                    )

    def test_sector_list_non_empty(self):
        for sym, entry in SPECIAL_SYMBOL_TERMS.items():
            self.assertTrue(entry.get("sector"), f"{sym} has empty sector list")


# ---------------------------------------------------------------------------
# 2. _get_search_terms() for special symbols
# ---------------------------------------------------------------------------

class TestGetSearchTermsSpecialSymbols(unittest.TestCase):

    def test_returns_display_name_for_index(self):
        result = _get_search_terms("^GSPC")
        self.assertEqual(result["display_name"], "S&P 500")

    def test_returns_display_name_for_futures(self):
        result = _get_search_terms("CL=F")
        self.assertEqual(result["display_name"], "Crude Oil")

    def test_names_do_not_contain_raw_caret_symbol(self):
        for sym in ["^GSPC", "^DJI", "^IXIC", "^VIX", "^RUT"]:
            result = _get_search_terms(sym)
            for name in result["names"]:
                self.assertNotIn("^", name, f"{sym}: name '{name}' contains '^'")

    def test_names_do_not_contain_equals_f(self):
        for sym in ["CL=F", "GC=F", "SI=F", "HG=F", "NG=F"]:
            result = _get_search_terms(sym)
            for name in result["names"]:
                self.assertNotIn("=F", name, f"{sym}: name '{name}' contains '=F'")

    def test_people_and_products_empty_for_special(self):
        for sym in ["^GSPC", "CL=F", "DX-Y.NYB"]:
            result = _get_search_terms(sym)
            self.assertEqual(result["people"], [])
            self.assertEqual(result["products"], [])

    def test_sector_non_empty_for_special(self):
        for sym in ["^DJI", "GC=F", "^TNX"]:
            result = _get_search_terms(sym)
            self.assertTrue(result["sector"], f"{sym}: sector is empty")

    def test_ticker_field_is_uppercase_original(self):
        result = _get_search_terms("^gspc")
        self.assertEqual(result["ticker"], "^GSPC")

    def test_composite_symbol_resolved(self):
        result = _get_search_terms("DX-Y.NYB")
        self.assertEqual(result["display_name"], "US Dollar Index")
        self.assertIn("DXY", result["names"])

    def test_all_known_special_symbols_resolve(self):
        for sym in SPECIAL_SYMBOL_TERMS:
            result = _get_search_terms(sym)
            self.assertTrue(result["names"], f"{sym}: names list is empty")
            self.assertIn("display_name", result)


# ---------------------------------------------------------------------------
# 3. Standard ticker regression — must be unaffected
# ---------------------------------------------------------------------------

class TestGetSearchTermsStandardTickers(unittest.TestCase):

    def test_aapl_returns_apple(self):
        result = _get_search_terms("AAPL")
        self.assertIn("Apple", result["names"])

    def test_tsla_has_people(self):
        result = _get_search_terms("TSLA")
        self.assertTrue(result["people"])

    def test_unknown_ticker_falls_back_to_symbol(self):
        result = _get_search_terms("ZZZZ")
        self.assertEqual(result["names"], ["ZZZZ"])

    def test_standard_ticker_has_display_name(self):
        result = _get_search_terms("MSFT")
        self.assertIn("display_name", result)
        self.assertTrue(result["display_name"])


# ---------------------------------------------------------------------------
# 4. NewsAPI query building — raw symbol must be absent for special symbols
# ---------------------------------------------------------------------------

class TestNewsApiQueryBuilding(unittest.TestCase):
    """
    Isolate the NewsAPI query construction logic by inspecting what
    _get_search_terms returns, then simulating the query-building code.
    """

    def _build_newsapi_query(self, ticker_symbol):
        """Mirror the query-building logic from iris_mvp.py analyze_news()."""
        search_terms = _get_search_terms(ticker_symbol)
        is_special = ticker_symbol in SPECIAL_SYMBOL_TERMS
        query_parts = [] if is_special else [f'"{ticker_symbol}"']
        for name in search_terms["names"][:3]:
            query_parts.append(f'"{name}"')
        for person in search_terms["people"][:1]:
            query_parts.append(f'"{person}"')
        return " OR ".join(query_parts)

    def test_gspc_query_has_no_caret(self):
        q = self._build_newsapi_query("^GSPC")
        self.assertNotIn("^GSPC", q)
        self.assertNotIn("^", q)

    def test_clf_query_has_no_equals_f(self):
        q = self._build_newsapi_query("CL=F")
        self.assertNotIn("CL=F", q)
        self.assertIn("crude oil", q.lower())

    def test_dji_query_contains_dow_jones(self):
        q = self._build_newsapi_query("^DJI")
        self.assertIn("Dow Jones", q)

    def test_gcf_query_contains_gold(self):
        q = self._build_newsapi_query("GC=F")
        self.assertIn("gold", q.lower())

    def test_standard_ticker_query_contains_raw_symbol(self):
        q = self._build_newsapi_query("AAPL")
        self.assertIn('"AAPL"', q)
        self.assertIn("Apple", q)

    def test_query_not_empty_for_any_special_symbol(self):
        for sym in SPECIAL_SYMBOL_TERMS:
            q = self._build_newsapi_query(sym)
            self.assertTrue(q.strip(), f"{sym}: NewsAPI query is empty")


# ---------------------------------------------------------------------------
# 5. Google News query building — name-based for special symbols
# ---------------------------------------------------------------------------

class TestGoogleNewsQueryBuilding(unittest.TestCase):

    def _build_gn_queries(self, ticker_symbol, lookback=21):
        """Mirror the Google News query-building logic from iris_mvp.py."""
        search_terms = _get_search_terms(ticker_symbol)
        primary_name = search_terms["names"][0] if search_terms.get("names") else ticker_symbol
        if ticker_symbol in SPECIAL_SYMBOL_TERMS:
            gn_queries = list(search_terms["names"][:2])
            if search_terms.get("sector"):
                gn_queries.append(search_terms["sector"][0])
        else:
            gn_queries = [f"{ticker_symbol} stock", primary_name]
            if lookback >= 60 and search_terms.get("sector"):
                gn_queries.append(search_terms["sector"][0])
        return gn_queries

    def test_gspc_queries_contain_no_caret(self):
        queries = self._build_gn_queries("^GSPC")
        for q in queries:
            self.assertNotIn("^", q, f"Query '{q}' contains raw '^'")

    def test_clf_queries_contain_no_equals(self):
        queries = self._build_gn_queries("CL=F")
        for q in queries:
            self.assertNotIn("=F", q, f"Query '{q}' contains raw '=F'")

    def test_special_symbol_gets_at_least_two_queries(self):
        for sym in ["^GSPC", "^DJI", "CL=F", "GC=F"]:
            queries = self._build_gn_queries(sym)
            self.assertGreaterEqual(len(queries), 2, f"{sym}: fewer than 2 GN queries")

    def test_standard_ticker_query_uses_raw_symbol(self):
        queries = self._build_gn_queries("TSLA")
        self.assertTrue(any("TSLA" in q for q in queries))


# ---------------------------------------------------------------------------
# 6. Simulation fallback uses display_name and name-based URLs
# ---------------------------------------------------------------------------

class TestSimulationFallback(unittest.TestCase):

    def _build_sim_items(self, ticker_symbol):
        """Mirror the simulation fallback else-branch from iris_mvp.py."""
        import urllib.parse as _sim_urlparse
        search_terms = _get_search_terms(ticker_symbol)
        _search_base = "https://news.google.com/search?q="
        _primary_name = search_terms["names"][0] if search_terms.get("names") else ticker_symbol
        _display = search_terms.get("display_name", _primary_name)
        _search_name = search_terms["names"][0] if search_terms.get("names") else _primary_name
        return [
            {"title": f"{_display} market update: latest price action and analysis",
             "url": _search_base + _sim_urlparse.quote(f"{_search_name} market news")},
            {"title": f"Analysts weigh in on {_display} outlook",
             "url": _search_base + _sim_urlparse.quote(f"{_search_name} analyst outlook")},
            {"title": f"Macro factors driving {_display} movement this week",
             "url": _search_base + _sim_urlparse.quote(f"{_search_name} macro factors")},
            {"title": f"Institutional activity in {_display}: what the data shows",
             "url": _search_base + _sim_urlparse.quote(f"{_search_name} institutional")},
            {"title": f"{_display} technical levels and trend analysis",
             "url": _search_base + _sim_urlparse.quote(f"{_search_name} technical analysis")},
        ]

    def test_gspc_sim_uses_sp500_not_caret(self):
        items = self._build_sim_items("^GSPC")
        for item in items:
            self.assertNotIn("^GSPC", item["title"])
            self.assertNotIn("^GSPC", item["url"])
            self.assertIn("S&P 500", item["title"])

    def test_clf_sim_uses_crude_oil(self):
        items = self._build_sim_items("CL=F")
        for item in items:
            self.assertNotIn("CL=F", item["title"])
            self.assertIn("Crude Oil", item["title"])

    def test_sim_urls_are_valid_google_news_links(self):
        for sym in ["^GSPC", "^DJI", "CL=F", "GC=F", "^VIX"]:
            items = self._build_sim_items(sym)
            for item in items:
                self.assertTrue(
                    item["url"].startswith("https://news.google.com/search?q="),
                    f"{sym}: URL '{item['url']}' is not a Google News search link"
                )

    def test_sim_produces_five_items(self):
        for sym in ["^GSPC", "CL=F", "DX-Y.NYB", "^TNX"]:
            items = self._build_sim_items(sym)
            self.assertEqual(len(items), 5, f"{sym}: expected 5 sim items")

    def test_standard_ticker_sim_uses_company_name(self):
        items = self._build_sim_items("AAPL")
        for item in items:
            self.assertNotIn("AAPL market update", item["title"])
            self.assertIn("Apple", item["title"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
