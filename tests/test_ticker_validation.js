'use strict';
/**
 * Unit tests for tickerValidation.js and the TickerInput component behaviour.
 * Run with:  node --test tests/test_ticker_validation.js
 * Requires:  npm install   (installs jsdom for DOM tests)
 */

const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { JSDOM } = require('jsdom');

// Load the validation module (UMD — sets module.exports in Node.js)
const { validateTickerFormat, validateTickerRemote } = require(
    path.join(__dirname, '..', 'static', 'tickerValidation.js')
);

// ---------------------------------------------------------------------------
// Pure format-validation tests (no DOM, no network)
// ---------------------------------------------------------------------------

describe('validateTickerFormat', () => {

    test('test_empty_input_shows_error — empty string returns error', () => {
        const result = validateTickerFormat('');
        assert.equal(result.valid, false);
        assert.match(result.error, /Please enter/i);
    });

    test('test_empty_input_shows_error — whitespace-only returns error', () => {
        const result = validateTickerFormat('   ');
        assert.equal(result.valid, false);
        assert.match(result.error, /Please enter/i);
    });

    test('test_numbers_rejected_instantly — digits in input', () => {
        for (const bad of ['123', '1A', 'A1B', '9XYZ']) {
            const result = validateTickerFormat(bad);
            assert.equal(result.valid, false,
                `Expected "${bad}" to be rejected`);
            assert.match(result.error, /invalid ticker format/i);
        }
    });

    test('test_too_long_ticker_rejected — 6+ letters rejected', () => {
        const result = validateTickerFormat('ABCDEF');
        assert.equal(result.valid, false);
        assert.match(result.error, /invalid ticker format/i);
    });

    test('reserved words rejected', () => {
        for (const word of ['TEST', 'NULL', 'NONE', 'HELP', 'NA']) {
            const result = validateTickerFormat(word);
            assert.equal(result.valid, false,
                `Expected "${word}" to be rejected as reserved`);
            assert.match(result.error, /not a stock ticker/i);
        }
    });

    test('test_valid_format_triggers_remote_check — valid formats pass', () => {
        for (const good of ['AAPL', 'msft', ' TSLA ', 'A', 'GOOGL']) {
            const result = validateTickerFormat(good);
            assert.equal(result.valid, true,
                `Expected "${good}" to pass format check`);
            assert.equal(result.cleaned, good.trim().toUpperCase());
        }
    });

    test('index symbols pass format check', () => {
        for (const sym of ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']) {
            const result = validateTickerFormat(sym);
            assert.equal(result.valid, true, `Expected "${sym}" to pass format check`);
        }
    });

    test('futures symbols pass format check', () => {
        for (const sym of ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F']) {
            const result = validateTickerFormat(sym);
            assert.equal(result.valid, true, `Expected "${sym}" to pass format check`);
        }
    });

    test('composite symbols pass format check', () => {
        const result = validateTickerFormat('DX-Y.NYB');
        assert.equal(result.valid, true, 'Expected "DX-Y.NYB" to pass format check');
    });

    test('invalid special-looking inputs still rejected', () => {
        for (const bad of ['^', '=F', 'CL=X', '^^DJI', 'TOOLONGBASE=F']) {
            const result = validateTickerFormat(bad);
            assert.equal(result.valid, false, `Expected "${bad}" to be rejected`);
        }
    });
});

// ---------------------------------------------------------------------------
// DOM behaviour tests (requires jsdom)
// ---------------------------------------------------------------------------

/** Build a minimal DOM that mirrors the search section in index.html */
function buildDOM() {
    const dom = new JSDOM(`<!DOCTYPE html>
<html><body>
  <form id="analyze-form">
    <div class="search-box">
      <div class="input-wrapper" id="ticker-input-wrapper">
        <input type="text" id="ticker-input" />
        <button type="button" id="ticker-clear" class="ticker-clear hidden">×</button>
        <span id="ticker-val-indicator" class="ticker-val-indicator hidden"></span>
      </div>
      <button type="submit" id="analyze-btn" disabled>
        <span class="btn-text">Analyze Risk</span>
      </button>
    </div>
  </form>
  <div id="validation-hint" class="hidden">
    <span id="validation-msg"></span>
    <div id="suggestion-chips"></div>
  </div>
  <div id="error-message" class="error-msg hidden"></div>
</body></html>`, { runScripts: 'dangerously', pretendToBeVisual: true });

    const { window } = dom;

    // Expose TickerValidation in the fake window before loading app logic
    window.TickerValidation = { validateTickerFormat, validateTickerRemote };

    return { window, document: window.document };
}

/**
 * Wire up a minimal version of the validation logic from app.js so we can
 * test component behaviour without importing the full 1 000-line app.js
 * (which has heavyweight dependencies like LightweightCharts).
 */
function wireValidation(window) {
    const { document } = window;
    const input         = document.getElementById('ticker-input');
    const analyzeBtn    = document.getElementById('analyze-btn');
    const clearBtn      = document.getElementById('ticker-clear');
    const validationHint  = document.getElementById('validation-hint');
    const validationMsgEl = document.getElementById('validation-msg');
    const suggestionChips = document.getElementById('suggestion-chips');

    let _validatedTicker = null;
    let _debounceTimer   = null;
    let _abortController = null;

    function getValidatedTicker() { return _validatedTicker; }

    function _setInputState(state) {
        input.className = state ? `ticker-input--${state}` : '';
    }

    function _showHint(text, type) {
        validationMsgEl.textContent = text;
        validationMsgEl.className = `validation-msg validation-msg--${type}`;
        validationHint.classList.remove('hidden');
    }

    function _clearHint() {
        validationHint.classList.add('hidden');
        validationMsgEl.textContent = '';
        suggestionChips.innerHTML = '';
    }

    function _renderSuggestions(suggestions) {
        suggestionChips.innerHTML = '';
        if (!Array.isArray(suggestions)) return;
        suggestions.forEach((s) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'suggestion-chip';
            chip.textContent = s;
            chip.addEventListener('click', () => {
                input.value = s;
                triggerValidation(s);
            });
            suggestionChips.appendChild(chip);
        });
    }

    async function triggerValidation(rawValue) {
        const val = String(rawValue || '').trim().toUpperCase();
        if (_abortController) _abortController.abort();
        _abortController = new window.AbortController();

        const fmt = window.TickerValidation.validateTickerFormat(val);
        if (!fmt.valid) {
            _validatedTicker = null;
            analyzeBtn.disabled = true;
            _setInputState('error');
            _showHint(fmt.error, 'error');
            _renderSuggestions([]);
            return;
        }

        _validatedTicker = null;
        analyzeBtn.disabled = true;
        _setInputState('validating');
        _clearHint();

        const { signal } = _abortController;
        const result = await window.TickerValidation.validateTickerRemote(val, signal);
        if (!result) return;

        if (result.valid) {
            _validatedTicker = val;
            analyzeBtn.disabled = false;
            _setInputState('valid');
            _showHint(`✓ ${result.company_name || val}`, 'success');
            _renderSuggestions([]);
        } else {
            _setInputState('error');
            _showHint(result.error || 'Not found.', 'error');
            _renderSuggestions(result.suggestions || []);
        }
    }

    input.addEventListener('input', () => {
        input.value = input.value.toUpperCase();
        clearBtn.classList.toggle('hidden', !input.value);
        const val = input.value.trim();
        clearTimeout(_debounceTimer);

        const fmt = window.TickerValidation.validateTickerFormat(val);
        if (!fmt.valid) {
            if (_abortController) _abortController.abort();
            _validatedTicker = null;
            analyzeBtn.disabled = true;
            _setInputState(val ? 'error' : '');
            if (val) _showHint(fmt.error, 'error');
            else _clearHint();
            _renderSuggestions([]);
            return;
        }

        _validatedTicker = null;
        analyzeBtn.disabled = true;
        _setInputState('validating');
        _clearHint();
        _debounceTimer = setTimeout(() => triggerValidation(val), 500);
    });

    clearBtn.addEventListener('click', () => {
        input.value = '';
        _validatedTicker = null;
        analyzeBtn.disabled = true;
        clearTimeout(_debounceTimer);
        if (_abortController) _abortController.abort();
        _setInputState('');
        _clearHint();
        clearBtn.classList.add('hidden');
    });

    return { input, analyzeBtn, clearBtn, validationHint, validationMsgEl,
             suggestionChips, triggerValidation, getValidatedTicker };
}

// Helper: fire input event
function fireInput(input, value) {
    input.value = value;
    input.dispatchEvent(new input.ownerDocument.defaultView.Event('input', { bubbles: true }));
}

describe('TickerInput DOM behaviour', () => {

    test('test_submit_disabled_until_valid — button starts disabled', () => {
        const { document } = buildDOM();
        const btn = document.getElementById('analyze-btn');
        assert.equal(btn.disabled, true);
    });

    test('test_suggestion_click_fills_input — clicking chip fills input', async () => {
        const { window } = buildDOM();

        // Mock validateTickerRemote to return suggestions for a bad ticker
        window.TickerValidation.validateTickerRemote = async () => ({
            valid: false,
            error: 'Not found.',
            suggestions: ['AAPL', 'APD'],
        });

        const { input, suggestionChips, triggerValidation } = wireValidation(window);

        await triggerValidation('AAPL1'); // invalid format handled locally
        // Use a format-valid but server-rejected ticker to get suggestions
        // Bypass format check by directly calling with a valid-format ticker
        // Override validateTickerFormat temporarily
        const original = window.TickerValidation.validateTickerFormat;
        window.TickerValidation.validateTickerFormat = () => ({ valid: true, cleaned: 'AAPX' });
        await triggerValidation('AAPX');
        window.TickerValidation.validateTickerFormat = original;

        const chips = suggestionChips.querySelectorAll('.suggestion-chip');
        assert.ok(chips.length > 0, 'Expected suggestion chips to be rendered');

        // Click the first chip — it should fill the input
        chips[0].click();
        assert.equal(input.value, chips[0].textContent);
    });

    test('test_submit_disabled_until_valid — button enabled after valid remote result', async () => {
        const { window } = buildDOM();
        window.TickerValidation.validateTickerRemote = async () => ({
            valid: true,
            ticker: 'AAPL',
            company_name: 'Apple Inc.',
        });

        const { analyzeBtn, triggerValidation } = wireValidation(window);

        assert.equal(analyzeBtn.disabled, true, 'Should start disabled');
        await triggerValidation('AAPL');
        assert.equal(analyzeBtn.disabled, false, 'Should be enabled after valid result');
    });
});
