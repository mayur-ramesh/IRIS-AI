import io
import json
import os
import sys
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import iris_mvp


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestWebzKeyEnvParsing(unittest.TestCase):
    def test_parse_multi_value_env_supports_common_separators(self):
        raw = " key-one,\nkey-two ; key-one \r\n key-three "
        self.assertEqual(
            iris_mvp._parse_multi_value_env(raw),
            ["key-one", "key-two", "key-three"],
        )

    def test_load_webz_api_keys_combines_plural_and_legacy_envs(self):
        with patch.dict(
            os.environ,
            {
                "WEBZ_API_KEYS": "key-one,key-two",
                "WEBZ_API_KEY": "key-three",
            },
            clear=False,
        ):
            self.assertEqual(
                iris_mvp._load_webz_api_keys_from_env(),
                ["key-one", "key-two", "key-three"],
            )


class TestNewsKeyEnvParsing(unittest.TestCase):
    def test_load_news_api_keys_combines_plural_and_legacy_envs(self):
        with patch.dict(
            os.environ,
            {
                "NEWS_API_KEYS": "news-one;news-two",
                "NEWS_API_KEY": "news-three",
            },
            clear=False,
        ):
            self.assertEqual(
                iris_mvp._load_news_api_keys_from_env(),
                ["news-one", "news-two", "news-three"],
            )


class TestNewsKeyRotation(unittest.TestCase):
    def _build_system(self, keys, client_factory):
        with patch.object(iris_mvp, "FINBERT_ENABLED", False), \
             patch.object(iris_mvp, "NEWS_API_KEYS", list(keys)), \
             patch.object(iris_mvp, "NEWS_API_KEY", keys[0] if keys else None), \
             patch.object(iris_mvp, "WEBZ_API_KEYS", []), \
             patch.object(iris_mvp, "WEBZ_API_KEY", None), \
             patch.object(iris_mvp, "NewsApiClient", side_effect=client_factory), \
             patch.object(iris_mvp.IRIS_System, "merge_alias_reports", return_value=None):
            return iris_mvp.IRIS_System()

    def test_rate_limited_newsapi_key_falls_back_to_next_key(self):
        calls = []

        class FakeNewsApiClient:
            def __init__(self, api_key):
                self.api_key = api_key

            def get_everything(self, **kwargs):
                calls.append((self.api_key, kwargs.get("q"), kwargs.get("sort_by")))
                if self.api_key == "news-first":
                    raise iris_mvp.NewsAPIException(
                        {
                            "status": "error",
                            "code": "rateLimited",
                            "message": "You have exceeded your rate limit.",
                        }
                    )
                return {
                    "status": "ok",
                    "articles": [
                        {
                            "title": "Recovered headline",
                            "url": "https://example.com/newsapi",
                            "publishedAt": "2026-04-08T00:00:00Z",
                        }
                    ],
                }

        system = self._build_system(
            ["news-first", "news-second"],
            client_factory=lambda api_key: FakeNewsApiClient(api_key),
        )

        response = system._news_api_get_everything(
            q='"AAPL" OR "Apple"',
            language="en",
            sort_by="publishedAt",
            from_param="2026-04-01",
            page_size=100,
        )

        self.assertEqual(len(response["articles"]), 1)
        self.assertEqual([api_key for api_key, _, _ in calls], ["news-first", "news-second"])
        self.assertEqual(system.news_api_key, "news-second")


class TestWebzKeyRotation(unittest.TestCase):
    def _build_system(self, keys):
        with patch.object(iris_mvp, "FINBERT_ENABLED", False), \
             patch.object(iris_mvp, "NEWS_API_KEYS", []), \
             patch.object(iris_mvp, "NEWS_API_KEY", None), \
             patch.object(iris_mvp, "WEBZ_API_KEYS", list(keys)), \
             patch.object(iris_mvp, "WEBZ_API_KEY", keys[0] if keys else None), \
             patch.object(iris_mvp.IRIS_System, "merge_alias_reports", return_value=None):
            return iris_mvp.IRIS_System()

    def test_http_429_falls_back_to_next_key(self):
        system = self._build_system(["first-key", "second-key"])
        calls = []

        def fake_urlopen(request, timeout=8):
            calls.append(request.full_url)
            if len(calls) == 1:
                raise HTTPError(
                    request.full_url,
                    429,
                    "Too Many Requests",
                    hdrs=None,
                    fp=io.BytesIO(b'{"message":"Call limit reached"}'),
                )
            return _FakeResponse(
                {
                    "posts": [
                        {
                            "title": "Fallback headline",
                            "url": "https://example.com/news",
                            "published": "2026-04-08T00:00:00Z",
                        }
                    ]
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            posts = system._fetch_webz_posts('"AAPL" language:english', 123456789)

        self.assertEqual(len(posts), 1)
        self.assertIn("token=first-key", calls[0])
        self.assertIn("token=second-key", calls[1])
        self.assertEqual(system.webz_api_key, "second-key")

    def test_error_payload_with_quota_message_falls_back_to_next_key(self):
        system = self._build_system(["first-key", "second-key"])
        responses = [
            _FakeResponse({"status": "error", "message": "Daily quota exceeded"}),
            _FakeResponse({"posts": [{"title": "Recovered", "url": "https://example.com"}]}),
        ]

        with patch("urllib.request.urlopen", side_effect=responses):
            posts = system._fetch_webz_posts('"MSFT" language:english', 987654321)

        self.assertEqual(len(posts), 1)
        self.assertEqual(system.webz_api_key, "second-key")


if __name__ == "__main__":
    unittest.main()
