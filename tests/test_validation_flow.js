'use strict';
/**
 * Integration tests for the end-to-end validation → analysis flow.
 * Tests the frontend behaviour when the backend returns 200 / 422 / 429 / 5xx.
 *
 * Run with:  node --test tests/test_validation_flow.js
 */

const { test, describe, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { JSDOM } = require('jsdom');

const { validateTickerFormat, validateTickerRemote } = require(
    path.join(__dirname, '..', 'static', 'tickerValidation.js')
);

// ---------------------------------------------------------------------------
// Minimal DOM that mirrors the relevant parts of index.html
// ---------------------------------------------------------------------------
function buildDOM() {
    return new JSDOM(`<!DOCTYPE html><html><body>
  <form id="analyze-form">
    <div class="search-box">
      <div id="ticker-input-wrapper">
        <input id="ticker-input" type="text" />
        <button id="ticker-clear" class="hidden" type="button">×</button>
        <span id="ticker-val-indicator" class="hidden"></span>
      </div>
      <button id="analyze-btn" type="submit" disabled>
        <span class="btn-text">Analyze Risk</span>
      </button>
    </div>
  </form>
  <div id="validation-hint" class="hidden">
    <span id="validation-msg"></span>
    <div id="suggestion-chips"></div>
  </div>
  <div id="error-banner"   class="hidden"><div id="error-banner-body">
    <span id="error-banner-msg"></span>
    <div id="error-banner-chips"></div>
  </div>
    <button id="error-banner-retry"   class="hidden" type="button">Retry</button>
    <button id="error-banner-dismiss" type="button">×</button>
  </div>
  <div id="analysis-progress" class="hidden">
    <div class="progress-step" id="prog-step-1"></div>
    <div class="progress-step" id="prog-step-2"></div>
    <div class="progress-step" id="prog-step-3"></div>
    <div class="progress-step" id="prog-step-4"></div>
  </div>
  <div id="analysis-skeleton" class="hidden"></div>
  <section id="results-dashboard" class="dashboard hidden"></section>
  <div id="error-message" class="hidden"></div>
</body></html>`, { pretendToBeVisual: true });
}

// ---------------------------------------------------------------------------
// Wire the analysis-flow logic (mirrors app.js without the full module)
// ---------------------------------------------------------------------------
function wireFlow(window, mockFetch) {
    const { document } = window;
    window.TickerValidation = { validateTickerFormat, validateTickerRemote };
    window.fetch = mockFetch;

    const input            = document.getElementById('ticker-input');
    const analyzeBtn       = document.getElementById('analyze-btn');
    const btnText          = analyzeBtn.querySelector('.btn-text');
    const dashboard        = document.getElementById('results-dashboard');
    const errorBanner      = document.getElementById('error-banner');
    const errorBannerMsg   = document.getElementById('error-banner-msg');
    const errorBannerChips = document.getElementById('error-banner-chips');
    const errorBannerRetry = document.getElementById('error-banner-retry');
    const analysisSkeleton = document.getElementById('analysis-skeleton');
    const analysisProgress = document.getElementById('analysis-progress');

    let _retryTicker    = null;
    let _validatedTicker = null;
    let _progressTimers  = [];
    let _rateLimitTimer  = null;

    function _showErrorBanner(message, suggestions, showRetry) {
        if (errorBannerMsg) errorBannerMsg.textContent = message;
        if (errorBannerChips) {
            errorBannerChips.innerHTML = '';
            (suggestions || []).forEach((s) => {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'suggestion-chip';
                chip.textContent = s;
                chip.addEventListener('click', () => {
                    input.value = s;
                    // In real app this calls _triggerValidation(s).
                    // Here we emit a custom event so tests can observe it.
                    input.dispatchEvent(new window.CustomEvent('validation-triggered', { detail: s }));
                });
                errorBannerChips.appendChild(chip);
            });
        }
        if (errorBannerRetry) errorBannerRetry.classList.toggle('hidden', !showRetry);
        errorBanner.classList.remove('hidden');
        errorBanner.classList.add('is-visible');
    }

    function _hideErrorBanner() {
        errorBanner.classList.remove('is-visible');
        errorBanner.classList.add('hidden');
    }

    function _showProgress() {
        analysisProgress.classList.remove('hidden');
        // Advance immediately to step 1
        analysisProgress.querySelectorAll('.progress-step').forEach((el, i) => {
            el.className = i === 0 ? 'progress-step is-active' : 'progress-step';
        });
    }

    function _hideProgress() {
        _progressTimers.forEach((t) => clearTimeout(t));
        _progressTimers = [];
        analysisProgress.classList.add('hidden');
    }

    function _showSkeleton() { analysisSkeleton.classList.remove('hidden'); }
    function _hideSkeleton()  { analysisSkeleton.classList.add('hidden');  }

    function _startRateLimitCountdown(seconds) {
        analyzeBtn.disabled = true;
        const endTime = Date.now() + seconds * 1000;
        function tick() {
            const remaining = Math.ceil((endTime - Date.now()) / 1000);
            if (remaining <= 0) {
                if (btnText) btnText.textContent = 'Analyze Risk';
                analyzeBtn.disabled = false;
                return;
            }
            if (btnText) btnText.textContent = `Wait ${remaining}s…`;
            _rateLimitTimer = setTimeout(tick, 500);
        }
        tick();
    }

    async function runAnalysis(ticker) {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        _retryTicker = normalizedTicker;

        const timeoutCtrl = new window.AbortController();
        const timeoutId = setTimeout(() => timeoutCtrl.abort(), 30000);

        analyzeBtn.disabled = true;
        _hideErrorBanner();
        _showProgress();
        _showSkeleton();
        dashboard.classList.add('hidden');

        try {
            const response = await window.fetch(`/api/analyze?ticker=${normalizedTicker}`, {
                signal: timeoutCtrl.signal,
            });
            clearTimeout(timeoutId);

            if (response.status === 422) {
                const body = await response.json().catch(() => ({}));
                _showErrorBanner(body.error || 'Invalid ticker.', body.suggestions || [], false);
                return;
            }

            if (response.status === 429) {
                _showErrorBanner(
                    "You're sending requests too quickly. Please wait a moment and try again.",
                    [], false
                );
                _startRateLimitCountdown(10);
                return;
            }

            const data = await response.json();
            if (!response.ok) {
                _showErrorBanner(data.error || 'Something went wrong.', [], true);
                return;
            }

            // Success
            _hideSkeleton();
            _hideProgress();
            dashboard.classList.remove('hidden');
            _validatedTicker = normalizedTicker;
            analyzeBtn.disabled = false;

        } catch (err) {
            clearTimeout(timeoutId);
            dashboard.classList.add('hidden');
            if (err.name === 'AbortError') {
                _showErrorBanner('The analysis is taking longer than expected. Please try again.', [], true);
            } else {
                _showErrorBanner('Something went wrong on our end. Please try again in a few seconds.', [], true);
            }
        } finally {
            _hideSkeleton();
            _hideProgress();
        }
    }

    return { runAnalysis, input, analyzeBtn, btnText, dashboard, errorBanner,
             errorBannerMsg, errorBannerChips, errorBannerRetry };
}

// Helpers
function makeFetch(status, body) {
    return async () => ({
        status,
        ok: status >= 200 && status < 300,
        json: async () => body,
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('End-to-end validation flow', () => {

    test('test_valid_ticker_shows_analysis_result — 200 renders dashboard', async () => {
        const { window } = buildDOM();
        const { runAnalysis, dashboard, errorBanner } = wireFlow(window, makeFetch(200, {
            meta: { symbol: 'AAPL' },
            market: { predicted_price_next_session: 200, history: [] },
        }));

        await runAnalysis('AAPL');

        assert.ok(!dashboard.classList.contains('hidden'),
            'Dashboard should be visible after 200 response');
        assert.ok(errorBanner.classList.contains('hidden'),
            'Error banner should remain hidden after success');
    });

    test('test_invalid_ticker_shows_error_banner — 422 shows banner with message', async () => {
        const { window } = buildDOM();
        const { runAnalysis, errorBanner, errorBannerMsg, errorBannerChips } = wireFlow(
            window,
            makeFetch(422, {
                valid: false,
                error: 'Ticker "XYZQW" was not found. Please check the symbol and try again.',
                suggestions: ['XYZ', 'XYZX'],
            })
        );

        await runAnalysis('XYZQW');

        assert.ok(!errorBanner.classList.contains('hidden'),
            'Error banner should be visible after 422');
        assert.ok(errorBannerMsg.textContent.includes('not found'),
            'Banner should display the error message');
        const chips = errorBannerChips.querySelectorAll('.suggestion-chip');
        assert.equal(chips.length, 2, 'Should render suggestion chips');
    });

    test('test_rate_limit_shows_cooldown — 429 disables button with countdown', async () => {
        const { window } = buildDOM();
        const { runAnalysis, analyzeBtn, errorBanner, errorBannerMsg } = wireFlow(
            window,
            makeFetch(429, {})
        );

        await runAnalysis('AAPL');

        assert.ok(!errorBanner.classList.contains('hidden'),
            'Error banner should show on 429');
        assert.ok(errorBannerMsg.textContent.includes('too quickly'),
            'Message should mention rate limiting');
        assert.ok(analyzeBtn.disabled,
            'Submit button should be disabled during cooldown');
    });

    test('test_suggestion_click_in_banner_retriggers_validation — chip click fires event', async () => {
        const { window } = buildDOM();
        const { runAnalysis, input, errorBannerChips } = wireFlow(
            window,
            makeFetch(422, {
                valid: false,
                error: 'Not found.',
                suggestions: ['AAPL', 'APD'],
            })
        );

        let triggeredTicker = null;
        input.addEventListener('validation-triggered', (e) => {
            triggeredTicker = e.detail;
        });

        await runAnalysis('AAPX');

        const chip = errorBannerChips.querySelector('.suggestion-chip');
        assert.ok(chip, 'Suggestion chip should be rendered');
        chip.click();

        assert.equal(input.value, chip.textContent,
            'Input should be filled with the clicked suggestion');
        assert.equal(triggeredTicker, chip.textContent,
            'Clicking suggestion should trigger validation for that ticker');
    });
});
