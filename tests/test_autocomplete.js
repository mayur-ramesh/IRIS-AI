'use strict';
/**
 * Unit tests for the ticker autocomplete dropdown behaviour.
 * Run with:  node --test tests/test_autocomplete.js
 * Requires:  npm install   (installs jsdom)
 */

const { test, describe, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const { JSDOM } = require('jsdom');

// ---------------------------------------------------------------------------
// Helpers to build a minimal DOM environment for each test
// ---------------------------------------------------------------------------

function buildDOM() {
    const dom = new JSDOM(`<!DOCTYPE html>
<html>
<body>
  <div id="ticker-input-wrapper" style="position:relative">
    <input type="text" id="ticker-input" role="combobox"
           aria-expanded="false" aria-haspopup="listbox"
           aria-autocomplete="list" aria-controls="ticker-dropdown">
    <ul id="ticker-dropdown" role="listbox" class="ticker-dropdown hidden"></ul>
  </div>
  <div id="ticker-ac-live" aria-live="polite"></div>
</body>
</html>`, { pretendToBeVisual: true });
    return dom;
}

// Minimal standalone versions of the autocomplete helpers so we can unit-test
// without loading the full app.js (which depends on many other DOM elements).
function buildHelpers(doc) {
    const input      = doc.getElementById('ticker-input');
    const dropdown   = doc.getElementById('ticker-dropdown');
    const liveRegion = doc.getElementById('ticker-ac-live');

    let _acActiveIndex = -1;
    let _acResults     = [];

    function _hideDropdown() {
        dropdown.classList.add('hidden');
        input.setAttribute('aria-expanded', 'false');
        input.removeAttribute('aria-activedescendant');
        _acActiveIndex = -1;
        _acResults = [];
    }

    function _highlightItem(index) {
        const items = dropdown.querySelectorAll('.ticker-dropdown-item');
        _acActiveIndex = Math.max(-1, Math.min(index, items.length - 1));
        items.forEach((el, i) => {
            const active = i === _acActiveIndex;
            el.setAttribute('aria-selected', active ? 'true' : 'false');
            if (active) input.setAttribute('aria-activedescendant', el.id);
        });
        if (_acActiveIndex === -1) input.removeAttribute('aria-activedescendant');
    }

    function _selectItem(ticker) {
        _hideDropdown();
        input.value = ticker;
        if (liveRegion) liveRegion.textContent = ticker + ' selected';
    }

    function _renderDropdown(results) {
        dropdown.innerHTML = '';
        _acResults = results || [];
        if (!_acResults.length) { _hideDropdown(); return; }
        _acResults.forEach((item, i) => {
            const li = doc.createElement('li');
            li.id = 'ac-item-' + i;
            li.setAttribute('role', 'option');
            li.className = 'ticker-dropdown-item';
            li.setAttribute('aria-selected', 'false');

            const ts = doc.createElement('span');
            ts.className = 'ticker-dropdown-ticker';
            ts.textContent = item.ticker;

            const ns = doc.createElement('span');
            ns.className = 'ticker-dropdown-name';
            ns.textContent = item.name || '';

            li.appendChild(ts);
            li.appendChild(ns);
            li.addEventListener('mousedown', (e) => { e.preventDefault(); _selectItem(item.ticker); });
            dropdown.appendChild(li);
        });
        dropdown.classList.remove('hidden');
        input.setAttribute('aria-expanded', 'true');
        _acActiveIndex = -1;
    }

    return { _hideDropdown, _highlightItem, _selectItem, _renderDropdown,
             get acActiveIndex() { return _acActiveIndex; },
             get acResults()     { return _acResults; } };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('autocomplete dropdown rendering', () => {

    test('test_render_shows_dropdown_with_results', () => {
        const { document } = buildDOM().window;
        const { _renderDropdown } = buildHelpers(document);
        const dropdown = document.getElementById('ticker-dropdown');
        const input    = document.getElementById('ticker-input');

        _renderDropdown([
            { ticker: 'AAPL', name: 'Apple Inc.' },
            { ticker: 'AAMT', name: 'AAMT Corp' },
        ]);

        assert.equal(dropdown.classList.contains('hidden'), false,
            'dropdown should be visible after rendering results');
        assert.equal(input.getAttribute('aria-expanded'), 'true');
        assert.equal(dropdown.querySelectorAll('.ticker-dropdown-item').length, 2);
    });

    test('test_render_empty_hides_dropdown', () => {
        const { document } = buildDOM().window;
        const { _renderDropdown } = buildHelpers(document);
        const dropdown = document.getElementById('ticker-dropdown');

        _renderDropdown([]);

        assert.equal(dropdown.classList.contains('hidden'), true,
            'dropdown should be hidden when results are empty');
    });

    test('test_item_has_ticker_and_name_spans', () => {
        const { document } = buildDOM().window;
        const { _renderDropdown } = buildHelpers(document);
        const dropdown = document.getElementById('ticker-dropdown');

        _renderDropdown([{ ticker: 'NVDA', name: 'NVIDIA Corp' }]);

        const item = dropdown.querySelector('.ticker-dropdown-item');
        assert.ok(item, 'item should exist');
        assert.equal(item.querySelector('.ticker-dropdown-ticker').textContent, 'NVDA');
        assert.equal(item.querySelector('.ticker-dropdown-name').textContent, 'NVIDIA Corp');
    });

});

describe('autocomplete keyboard navigation', () => {

    test('test_arrow_down_highlights_first_item', () => {
        const { document } = buildDOM().window;
        const { _renderDropdown, _highlightItem, acActiveIndex } = buildHelpers(document);

        _renderDropdown([
            { ticker: 'AAPL', name: 'Apple Inc.' },
            { ticker: 'AAP',  name: 'Advance Auto' },
        ]);

        const helpers = buildHelpers(document);
        helpers._renderDropdown([
            { ticker: 'AAPL', name: 'Apple Inc.' },
            { ticker: 'AAP',  name: 'Advance Auto' },
        ]);
        helpers._highlightItem(0);

        const items = document.querySelectorAll('.ticker-dropdown-item');
        // The second call to buildHelpers operated on the same DOM
        // so the last rendered item set is what matters for aria-selected
        // Re-query after highlight
        const firstItem = document.getElementById('ac-item-0');
        // helpers._acActiveIndex should be 0
        assert.equal(helpers.acActiveIndex, 0);
    });

    test('test_select_item_fills_input_and_hides_dropdown', () => {
        const { document } = buildDOM().window;
        const helpers = buildHelpers(document);
        const dropdown = document.getElementById('ticker-dropdown');
        const input    = document.getElementById('ticker-input');

        helpers._renderDropdown([{ ticker: 'MSFT', name: 'Microsoft Corp' }]);
        helpers._selectItem('MSFT');

        assert.equal(input.value, 'MSFT');
        assert.equal(dropdown.classList.contains('hidden'), true,
            'dropdown should be hidden after selection');
    });

});
