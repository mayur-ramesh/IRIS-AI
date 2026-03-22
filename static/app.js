const THEME_STORAGE_KEY = 'iris-theme';

function getPreferredTheme() {
    let savedTheme = null;
    try {
        savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    } catch (error) {
        savedTheme = null;
    }

    if (savedTheme === 'light' || savedTheme === 'dark') {
        return savedTheme;
    }

    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return prefersDark ? 'dark' : 'light';
}

function applyTheme(theme) {
    const normalizedTheme = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', normalizedTheme);
}

(function initializeThemeImmediately() {
    applyTheme(getPreferredTheme());
})();

document.addEventListener('DOMContentLoaded', () => {
    const TIMEFRAME_TO_QUERY = {
        '1D': { period: '1d', interval: '2m' },
        '5D': { period: '5d', interval: '15m' },
        '1M': { period: '1mo', interval: '60m' },
        '6M': { period: '6mo', interval: '1d' },
        'YTD': { period: 'ytd', interval: '1d' },
        '1Y': { period: '1y', interval: '1d' },
        '5Y': { period: '5y', interval: '1wk' },
    };

    const themeToggle = document.getElementById('theme-toggle');
    const timeframeButtons = Array.from(document.querySelectorAll('.timeframe-btn'));
    const form = document.getElementById('analyze-form');
    const input = document.getElementById('ticker-input');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.getElementById('loading-spinner');
    const analyzeBtn = document.getElementById('analyze-btn');
    const errorMsg = document.getElementById('error-message');

    // Dashboard Elements
    const dashboard = document.getElementById('results-dashboard');
    const resTicker = document.getElementById('res-ticker');
    const resTime = document.getElementById('res-time');

    const engineIndicator = document.getElementById('engine-light');
    const lightStatusText = document.getElementById('light-status');

    const currentPriceEl = document.getElementById('current-price');
    const predictedPriceEl = document.getElementById('predicted-price');
    const trendLabelEl = document.getElementById('trend-label');

    const sentimentScoreEl = document.getElementById('sentiment-score');
    const sentimentDescEl = document.getElementById('sentiment-desc');
    const headlinesList = document.getElementById('headlines-list');
    const feedbackOpenBtn = document.getElementById('feedback-open');
    const feedbackModal = document.getElementById('feedback-modal');
    const feedbackCancelBtn = document.getElementById('feedback-cancel');
    const feedbackSubmitBtn = document.getElementById('feedback-submit');
    const feedbackMessageEl = document.getElementById('feedback-message');

    const chartContainer = document.getElementById('advanced-chart');
    const chartPlaceholder = document.getElementById('chart-placeholder');
    let lwChart = null;
    let currentTicker = '';
    let latestPredictedPrice = null;
    let latestAnalyzeHistory = [];
    let latestAnalyzeTimeframe = '6M';
    // --- Analysis state / flow elements ---
    const errorBanner       = document.getElementById('error-banner');
    const errorBannerMsg    = document.getElementById('error-banner-msg');
    const errorBannerChips  = document.getElementById('error-banner-chips');
    const errorBannerRetry  = document.getElementById('error-banner-retry');
    const errorBannerDismiss= document.getElementById('error-banner-dismiss');
    const analysisProgress  = document.getElementById('analysis-progress');
    const analysisSkeleton  = document.getElementById('analysis-skeleton');

    let _retryTicker      = null;
    let _progressTimers   = [];
    let _rateLimitTimer   = null;

    function _showErrorBanner(message, suggestions, showRetry, bannerType) {
        if (!errorBanner) return;
        // Apply visual variant: 'error' (red, default) | 'warning' (yellow) | 'muted' (gray)
        errorBanner.dataset.bannerType = bannerType || 'error';
        if (errorBannerMsg) errorBannerMsg.textContent = message;
        if (errorBannerChips) {
            errorBannerChips.innerHTML = '';
            if (Array.isArray(suggestions) && suggestions.length) {
                suggestions.forEach((s) => {
                    const chip = document.createElement('button');
                    chip.type = 'button';
                    chip.className = 'suggestion-chip';
                    chip.textContent = s;
                    chip.addEventListener('click', () => {
                        _hideErrorBanner();
                        input.value = s;
                        _triggerValidation(s);
                    });
                    errorBannerChips.appendChild(chip);
                });
            }
        }
        if (errorBannerRetry) errorBannerRetry.classList.toggle('hidden', !showRetry);
        errorBanner.classList.remove('hidden');
        requestAnimationFrame(() => errorBanner.classList.add('is-visible'));
    }

    function _hideErrorBanner() {
        if (!errorBanner) return;
        errorBanner.classList.remove('is-visible');
        setTimeout(() => errorBanner.classList.add('hidden'), 300);
    }

    function _clearProgressTimers() {
        _progressTimers.forEach((t) => clearTimeout(t));
        _progressTimers = [];
    }

    function _advanceProgress(step) {
        if (!analysisProgress) return;
        analysisProgress.querySelectorAll('.progress-step').forEach((el, i) => {
            const n = i + 1;
            el.className = n < step ? 'progress-step is-done'
                         : n === step ? 'progress-step is-active'
                         : 'progress-step';
        });
    }

    function _showProgress() {
        if (!analysisProgress) return;
        _advanceProgress(1);
        analysisProgress.classList.remove('hidden');
        _progressTimers.push(setTimeout(() => _advanceProgress(2), 1000));
        _progressTimers.push(setTimeout(() => _advanceProgress(3), 3000));
        _progressTimers.push(setTimeout(() => _advanceProgress(4), 5000));
    }

    function _hideProgress() {
        _clearProgressTimers();
        if (!analysisProgress) return;
        analysisProgress.classList.add('hidden');
        analysisProgress.querySelectorAll('.progress-step').forEach((el) => {
            el.className = 'progress-step';
        });
    }

    function _showSkeleton() { if (analysisSkeleton) analysisSkeleton.classList.remove('hidden'); }
    function _hideSkeleton()  { if (analysisSkeleton) analysisSkeleton.classList.add('hidden');  }

    function _startRateLimitCountdown(seconds) {
        const endTime = Date.now() + seconds * 1000;
        analyzeBtn.disabled = true;
        function tick() {
            const remaining = Math.ceil((endTime - Date.now()) / 1000);
            if (remaining <= 0) {
                if (btnText) btnText.textContent = 'Analyze Risk';
                analyzeBtn.disabled = !_validatedTicker;
                return;
            }
            if (btnText) btnText.textContent = `Wait ${remaining}s…`;
            _rateLimitTimer = setTimeout(tick, 500);
        }
        tick();
    }

    if (errorBannerDismiss) {
        errorBannerDismiss.addEventListener('click', _hideErrorBanner);
    }
    if (errorBannerRetry) {
        errorBannerRetry.addEventListener('click', () => {
            if (_retryTicker) {
                _hideErrorBanner();
                loadTickerData(_retryTicker, false);
            }
        });
    }

    // --- Validation UI elements ---
    const inputWrapper    = document.getElementById('ticker-input-wrapper');
    const clearBtn        = document.getElementById('ticker-clear');
    const valIndicator    = document.getElementById('ticker-val-indicator');
    const validationHint  = document.getElementById('validation-hint');
    const validationMsgEl = document.getElementById('validation-msg');
    const suggestionChips = document.getElementById('suggestion-chips');

    // --- Validation state ---
    let _validatedTicker  = null;   // non-null only when remote validation passed
    let _debounceTimer    = null;
    let _abortController  = null;

    function _setInputState(state) {
        // state: '' | 'error' | 'validating' | 'valid' | 'warn'
        input.className = state ? `ticker-input--${state}` : '';
        if (valIndicator) {
            valIndicator.classList.toggle('is-spinning', state === 'validating');
            valIndicator.classList.toggle('hidden', state !== 'validating');
        }
    }

    function _showValidationHint(text, type) {
        if (!validationMsgEl || !validationHint) return;
        validationMsgEl.textContent = text;
        validationMsgEl.className = `validation-msg validation-msg--${type}`;
        validationHint.classList.remove('hidden');
    }

    function _clearValidationHint() {
        if (validationHint) validationHint.classList.add('hidden');
        if (validationMsgEl) validationMsgEl.textContent = '';
        if (suggestionChips) suggestionChips.innerHTML = '';
    }

    function _renderSuggestions(suggestions) {
        if (!suggestionChips) return;
        suggestionChips.innerHTML = '';
        if (!Array.isArray(suggestions) || !suggestions.length) return;
        suggestions.forEach((s) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'suggestion-chip';
            chip.textContent = s;
            chip.addEventListener('click', () => {
                input.value = s;
                _triggerValidation(s);
            });
            suggestionChips.appendChild(chip);
        });
    }

    // Route a failed remote-validation result to the right visual treatment
    function _routeValidationError(result, val) {
        const code = result.code || '';
        const err = result.error || 'Validation failed.';
        if (code) console.debug('[IRIS-AI] Validation rejected — code:', code, '— ticker:', val);

        if (code === 'API_TIMEOUT' || code === 'API_ERROR') {
            // Service degraded: yellow warning banner with retry option
            _setInputState('warn');
            _clearValidationHint();
            _showErrorBanner(err, [], true, 'warning');
        } else if (code === 'TICKER_NOT_FOUND' || code === 'TICKER_DELISTED') {
            // Not found / delisted: error banner with suggestions
            _setInputState('error');
            _clearValidationHint();
            _showErrorBanner(err, result.suggestions || [], false);
        } else if (code === 'RATE_LIMITED') {
            // Rate limited: gray countdown banner
            _setInputState('');
            _clearValidationHint();
            const match = err.match(/(\d+)/);
            _startRateLimitCountdown(match ? parseInt(match[1], 10) : 30);
        } else {
            // Format / reserved-word / unknown: inline hint below input
            _setInputState('error');
            _showValidationHint(err, 'error');
            _renderSuggestions(result.suggestions || []);
        }
    }

    async function _triggerValidation(rawValue) {
        // Sanitise input first (mirrors Python sanitize_ticker_input)
        const sanitize = (window.TickerValidation || {}).sanitizeTicker;
        const val = sanitize ? sanitize(rawValue) : String(rawValue || '').trim().toUpperCase();

        // Sync input field to sanitised form
        if (input && input.value !== val && val) input.value = val;

        // Cancel previous in-flight remote request
        if (_abortController) _abortController.abort();
        _abortController = new AbortController();

        // Instant format check
        const fmt = (window.TickerValidation || {}).validateTickerFormat;
        if (!fmt) return;
        const fmtResult = fmt(val);
        if (!fmtResult.valid) {
            _validatedTicker = null;
            analyzeBtn.disabled = true;
            if (fmtResult.code) console.debug('[IRIS-AI] Format check failed — code:', fmtResult.code, '— input:', val);
            _setInputState('error');
            _showValidationHint(fmtResult.error, 'error');
            _renderSuggestions([]);
            if (clearBtn) clearBtn.classList.toggle('hidden', !val);
            return;
        }

        // Format OK → remote check
        _validatedTicker = null;
        analyzeBtn.disabled = true;
        _setInputState('validating');
        _clearValidationHint();
        if (clearBtn) clearBtn.classList.remove('hidden');

        const { signal } = _abortController;
        const remoteCheck = (window.TickerValidation || {}).validateTickerRemote;
        if (!remoteCheck) return;

        const result = await remoteCheck(val, signal);
        if (!result) return; // cancelled by a newer keystroke

        if (result.valid) {
            _validatedTicker = val;
            analyzeBtn.disabled = false;
            const hasWarning = !!(result.warning);
            _setInputState(hasWarning ? 'warn' : 'valid');
            _showValidationHint(
                hasWarning ? `\u26A0 ${result.warning}` : `\u2713 ${result.company_name || val}`,
                hasWarning ? 'warn' : 'success'
            );
            _renderSuggestions([]);
        } else {
            _routeValidationError(result, val);
        }
    }

    if (input) {
        input.addEventListener('input', () => {
            // Auto-uppercase + strip leading $ / # as user types
            let pos = input.selectionStart || 0;
            let v = input.value.toUpperCase().replace(/^[\$#]+/, '');
            if (v !== input.value) {
                const removed = input.value.length - v.length;
                input.value = v;
                pos = Math.max(0, pos - removed);
            }
            try { input.setSelectionRange(pos, pos); } catch (_) {}

            if (clearBtn) clearBtn.classList.toggle('hidden', !input.value);

            const val = input.value.trim();
            clearTimeout(_debounceTimer);

            // Instant format check for immediate feedback
            const fmt = (window.TickerValidation || {}).validateTickerFormat;
            if (fmt) {
                const fmtResult = fmt(val);
                if (!fmtResult.valid) {
                    if (_abortController) _abortController.abort();
                    _validatedTicker = null;
                    analyzeBtn.disabled = true;
                    _setInputState(val ? 'error' : '');
                    if (val) {
                        _showValidationHint(fmtResult.error, 'error');
                    } else {
                        _clearValidationHint();
                        _setInputState('');
                    }
                    _renderSuggestions([]);
                    return;
                }
            }

            // Format OK — debounce the remote call
            _validatedTicker = null;
            analyzeBtn.disabled = true;
            _setInputState('validating');
            _clearValidationHint();
            _debounceTimer = setTimeout(() => _triggerValidation(val), 500);
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            input.value = '';
            _validatedTicker = null;
            analyzeBtn.disabled = true;
            clearTimeout(_debounceTimer);
            if (_abortController) _abortController.abort();
            _setInputState('');
            _clearValidationHint();
            clearBtn.classList.add('hidden');
            input.focus();
        });
    }
    // --- End ticker validation ---

    // --- Ticker autocomplete ---
    const dropdown      = document.getElementById('ticker-dropdown');
    const acLiveRegion  = document.getElementById('ticker-ac-live');

    let _acDebounceTimer   = null;
    let _acAbortController = null;
    let _acActiveIndex     = -1;
    let _acResults         = [];

    function _hideDropdown() {
        if (!dropdown) return;
        dropdown.classList.add('hidden');
        if (input) input.setAttribute('aria-expanded', 'false');
        if (input) input.removeAttribute('aria-activedescendant');
        _acActiveIndex = -1;
        _acResults = [];
    }

    function _highlightItem(index) {
        if (!dropdown) return;
        const items = dropdown.querySelectorAll('.ticker-dropdown-item');
        _acActiveIndex = Math.max(-1, Math.min(index, items.length - 1));
        items.forEach((el, i) => {
            const active = i === _acActiveIndex;
            el.setAttribute('aria-selected', active ? 'true' : 'false');
            if (active) {
                el.scrollIntoView({ block: 'nearest' });
                if (input) input.setAttribute('aria-activedescendant', el.id);
            }
        });
        if (_acActiveIndex === -1 && input) input.removeAttribute('aria-activedescendant');
    }

    function _selectItem(ticker) {
        _hideDropdown();
        if (input) input.value = ticker;
        _triggerValidation(ticker);
        if (acLiveRegion) acLiveRegion.textContent = ticker + ' selected';
    }

    function _renderDropdown(results) {
        if (!dropdown) return;
        dropdown.innerHTML = '';
        _acResults = results || [];
        if (!_acResults.length) {
            _hideDropdown();
            return;
        }
        _acResults.forEach((item, i) => {
            const li = document.createElement('li');
            li.id = 'ac-item-' + i;
            li.setAttribute('role', 'option');
            li.className = 'ticker-dropdown-item';
            li.setAttribute('aria-selected', 'false');

            const tickerSpan = document.createElement('span');
            tickerSpan.className = 'ticker-dropdown-ticker';
            tickerSpan.textContent = item.ticker;

            const nameSpan = document.createElement('span');
            nameSpan.className = 'ticker-dropdown-name';
            nameSpan.textContent = item.name || '';

            li.appendChild(tickerSpan);
            li.appendChild(nameSpan);

            // mousedown prevents blur before click registers
            li.addEventListener('mousedown', (e) => {
                e.preventDefault();
                _selectItem(item.ticker);
            });

            dropdown.appendChild(li);
        });
        dropdown.classList.remove('hidden');
        if (input) input.setAttribute('aria-expanded', 'true');
        _acActiveIndex = -1;
    }

    async function _fetchAutocomplete(query) {
        if (_acAbortController) _acAbortController.abort();
        _acAbortController = new AbortController();
        try {
            const resp = await fetch(
                '/api/tickers/search?q=' + encodeURIComponent(query) + '&limit=8',
                { signal: _acAbortController.signal }
            );
            if (!resp.ok) { _hideDropdown(); return; }
            const data = await resp.json();
            _renderDropdown(data.results || []);
        } catch (err) {
            if (err.name !== 'AbortError') _hideDropdown();
        }
    }

    if (input) {
        // Autocomplete on input — 200 ms debounce
        input.addEventListener('input', () => {
            clearTimeout(_acDebounceTimer);
            const q = input.value.trim();
            if (q.length < 1) { _hideDropdown(); return; }
            _acDebounceTimer = setTimeout(() => _fetchAutocomplete(q), 200);
        });

        // Keyboard navigation inside the dropdown
        input.addEventListener('keydown', (e) => {
            if (!dropdown || dropdown.classList.contains('hidden')) return;
            const items = dropdown.querySelectorAll('.ticker-dropdown-item');
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                _highlightItem(Math.min(_acActiveIndex + 1, items.length - 1));
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                _highlightItem(Math.max(_acActiveIndex - 1, 0));
            } else if (e.key === 'Enter' && _acActiveIndex >= 0) {
                e.preventDefault();
                _selectItem(_acResults[_acActiveIndex].ticker);
            } else if (e.key === 'Escape') {
                _hideDropdown();
            }
        });

        // Hide when focus leaves the input
        input.addEventListener('blur', () => {
            setTimeout(_hideDropdown, 150);
        });
    }

    // Hide dropdown on click outside
    document.addEventListener('click', (e) => {
        if (dropdown && !dropdown.contains(e.target) && e.target !== input) {
            _hideDropdown();
        }
    });
    // --- End ticker autocomplete ---

    let historyRequestId = 0;
    const usdFormatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });

    const headlineDateFormatter = new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    });

    function formatHeadlineDate(raw) {
      if (!raw) return '';
      let d;
      const num = Number(raw);
      if (!isNaN(num) && num > 1e9) {
        d = new Date(num * 1000);   // Unix seconds → ms
      } else {
        d = new Date(raw);
      }
      if (isNaN(d.getTime())) return '';
      return headlineDateFormatter.format(d);
    }

    function extractDomain(url) {
      if (!url) return '';
      try {
        return new URL(url).hostname.replace(/^www\./, '');
      } catch {
        return '';
      }
    }

    const chartTooltip = document.createElement('div');
    chartTooltip.className = 'chart-hover-tooltip';
    chartTooltip.style.position = 'absolute';
    chartTooltip.style.pointerEvents = 'none';
    chartTooltip.style.display = 'none';
    chartTooltip.style.zIndex = '12';
    chartTooltip.style.padding = '8px 10px';
    chartTooltip.style.borderRadius = '8px';
    chartTooltip.style.border = '1px solid var(--panel-border)';
    chartTooltip.style.background = 'var(--panel-bg)';
    chartTooltip.style.color = 'var(--text-main)';
    chartTooltip.style.fontSize = '0.82rem';
    chartTooltip.style.fontWeight = '600';
    chartTooltip.style.lineHeight = '1.35';
    if (chartContainer) {
        chartContainer.appendChild(chartTooltip);
        chartContainer.addEventListener('mouseleave', () => {
            chartTooltip.style.display = 'none';
        });
    }

    function getChartDimensions() {
        if (!chartContainer) {
            return { width: 0, height: 300 };
        }
        const styles = window.getComputedStyle(chartContainer);
        const padX = (parseFloat(styles.paddingLeft) || 0) + (parseFloat(styles.paddingRight) || 0);
        const padY = (parseFloat(styles.paddingTop) || 0) + (parseFloat(styles.paddingBottom) || 0);
        const width = Math.max(0, chartContainer.clientWidth - padX);
        const rawHeight = chartContainer.clientHeight - padY;
        const height = Math.max(220, Number.isFinite(rawHeight) ? rawHeight : 300);
        return { width, height };
    }

    function resizeChartToContainer() {
        if (!lwChart || !chartContainer) return;
        const { width, height } = getChartDimensions();
        if (!width || !height) return;
        if (typeof lwChart.resize === 'function') {
            lwChart.resize(width, height);
        } else {
            lwChart.applyOptions({ width, height });
        }
    }

    function syncThemeToggleState() {
        if (!themeToggle) return;
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        themeToggle.setAttribute('aria-pressed', isDark ? 'true' : 'false');
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
            const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(nextTheme);
            try {
                localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
            } catch (error) {
                // Ignore storage failures and continue with in-memory theme.
            }
            syncThemeToggleState();
        });
        syncThemeToggleState();
    }

    function getActiveTimeframe() {
        const activeBtn = timeframeButtons.find((btn) => btn.classList.contains('active'));
        const key = String(activeBtn?.dataset?.timeframe || '6M').toUpperCase();
        return TIMEFRAME_TO_QUERY[key] ? key : '6M';
    }

    function setActiveTimeframe(timeframeKey) {
        const normalized = String(timeframeKey || '').toUpperCase();
        timeframeButtons.forEach((btn) => {
            const btnKey = String(btn.dataset.timeframe || '').toUpperCase();
            btn.classList.toggle('active', btnKey === normalized);
        });
    }

    function resolveTimeframeFromMeta(meta) {
        const period = String(meta?.period || '').toLowerCase();
        const interval = String(meta?.interval || '').toLowerCase();
        const match = Object.entries(TIMEFRAME_TO_QUERY).find(
            ([, value]) => value.period === period && value.interval === interval
        );
        return match ? match[0] : getActiveTimeframe();
    }

    async function fetchHistoryData(ticker, timeframeKey) {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        if (!normalizedTicker) return [];
        const activeTimeframe = TIMEFRAME_TO_QUERY[timeframeKey] ? timeframeKey : '6M';
        const mapped = TIMEFRAME_TO_QUERY[activeTimeframe];
        const params = new URLSearchParams({
            period: mapped.period,
            interval: mapped.interval,
        });
        const response = await fetch(`/api/history/${encodeURIComponent(normalizedTicker)}?${params.toString()}`);
        const body = await response.json();
        if (!response.ok) {
            throw new Error(body.error || 'Failed to fetch history data');
        }
        const history = normalizeHistoryPoints(body.data);
        if (history.length > 0) {
            return history;
        }
        const providerMessage = String(body?.message || '').trim();
        throw new Error(
            providerMessage || `No history data returned for ${normalizedTicker} (${mapped.period}, ${mapped.interval}).`
        );
    }

    async function fetchAnalyzeHistoryData(ticker, timeframeKey) {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        if (!normalizedTicker) return { history: [], predicted: null };
        const activeTimeframe = TIMEFRAME_TO_QUERY[timeframeKey] ? timeframeKey : '6M';
        const mapped = TIMEFRAME_TO_QUERY[activeTimeframe];

        const params = new URLSearchParams({ ticker: normalizedTicker });
        params.set('timeframe', activeTimeframe);
        params.set('period', mapped.period);
        params.set('interval', mapped.interval);

        const response = await fetch(`/api/analyze?${params.toString()}`);
        const body = await response.json();
        if (!response.ok) {
            throw new Error(body.error || 'Failed to fetch fallback history data');
        }

        const history = normalizeHistoryPoints(body?.market?.history);
        if (!history.length) {
            throw new Error(`No fallback history data returned for ${normalizedTicker} (${mapped.period}, ${mapped.interval}).`);
        }
        const predicted = Number(body?.market?.predicted_price_next_session);
        return { history, predicted: Number.isFinite(predicted) ? predicted : null };
    }

    async function refreshChartForTimeframe(ticker, timeframeKey, useLoadingState = true) {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        if (!normalizedTicker) return;
        const requestedTimeframe = TIMEFRAME_TO_QUERY[timeframeKey] ? timeframeKey : '6M';
        const requestId = ++historyRequestId;
        if (useLoadingState) {
            setLoading(true);
        }
        try {
            const history = await fetchHistoryData(normalizedTicker, requestedTimeframe);
            if (requestId !== historyRequestId) return;
            latestAnalyzeHistory = history;
            latestAnalyzeTimeframe = requestedTimeframe;
            renderChart(history, latestPredictedPrice);
        } catch (error) {
            if (requestId !== historyRequestId) return;
            console.error(error);
            try {
                const fallback = await fetchAnalyzeHistoryData(normalizedTicker, requestedTimeframe);
                if (requestId !== historyRequestId) return;
                latestAnalyzeHistory = fallback.history;
                latestAnalyzeTimeframe = requestedTimeframe;
                if (Number.isFinite(fallback.predicted)) {
                    latestPredictedPrice = fallback.predicted;
                }
                renderChart(fallback.history, latestPredictedPrice);
                errorMsg.classList.add('hidden');
                return;
            } catch (fallbackError) {
                if (requestId !== historyRequestId) return;
                console.error(fallbackError);
            }

            if (Array.isArray(latestAnalyzeHistory) && latestAnalyzeHistory.length > 0) {
                renderChart(latestAnalyzeHistory, latestPredictedPrice);
                errorMsg.classList.add('hidden');
                return;
            }

            if (chartContainer) {
                chartPlaceholder.classList.remove('hidden');
                chartPlaceholder.textContent = "Chart not available.";
                chartTooltip.style.display = 'none';
            }
            errorMsg.textContent = error.message || 'Failed to fetch chart data';
            errorMsg.classList.remove('hidden');
        } finally {
            if (useLoadingState) {
                setLoading(false);
            }
        }
    }

    async function loadTickerData(ticker, keepDashboardVisible = false) {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        if (!normalizedTicker) return;
        const activeTimeframe = getActiveTimeframe();
        const mapped = TIMEFRAME_TO_QUERY[activeTimeframe];
        const params = new URLSearchParams({ ticker: normalizedTicker });
        params.set('timeframe', activeTimeframe);
        params.set('period', mapped.period);
        params.set('interval', mapped.interval);

        _retryTicker = normalizedTicker;

        // 30-second hard timeout on the entire analysis
        const timeoutCtrl = new AbortController();
        const timeoutId = setTimeout(() => timeoutCtrl.abort(), 30000);

        setLoading(true);
        _hideErrorBanner();
        errorMsg.classList.add('hidden');
        _showProgress();
        _showSkeleton();
        if (!keepDashboardVisible) {
            dashboard.classList.add('hidden');
            var recSec = document.getElementById('recommended-section');
            if (recSec) recSec.classList.add('hidden');
        }

        try {
            const response = await fetch(`/api/analyze?${params.toString()}`, {
                signal: timeoutCtrl.signal,
            });
            clearTimeout(timeoutId);

            if (response.status === 422) {
                const body = await response.json().catch(() => ({}));
                _showErrorBanner(
                    body.error || 'Invalid ticker. Please check the symbol.',
                    body.suggestions || [],
                    false
                );
                _setInputState('error');
                _validatedTicker = null;
                analyzeBtn.disabled = true;
                return;
            }

            if (response.status === 429) {
                _showErrorBanner(
                    "You're sending requests too quickly. Please wait a moment and try again.",
                    [],
                    false
                );
                _startRateLimitCountdown(10);
                return;
            }

            const data = await response.json();
            if (!response.ok) {
                _showErrorBanner(
                    data.error || 'Something went wrong on our end. Please try again in a few seconds.',
                    [],
                    true
                );
                return;
            }

            // Success
            dashboard.classList.remove('hidden');
            currentTicker = String(data?.meta?.symbol || normalizedTicker).toUpperCase();
            input.value = currentTicker;
            _validatedTicker = currentTicker;
            latestPredictedPrice = Number(data?.market?.predicted_price_next_session);
            latestAnalyzeHistory = normalizeHistoryPoints(data?.market?.history);
            latestAnalyzeTimeframe = getActiveTimeframe();
            updateDashboard(data);
            await refreshChartForTimeframe(currentTicker, getActiveTimeframe(), false);
            if (typeof window._irisLoadRecommendations === 'function') { window._irisLoadRecommendations(currentTicker); }

        } catch (error) {
            clearTimeout(timeoutId);
            console.error(error);
            if (!keepDashboardVisible) {
                dashboard.classList.add('hidden');
            }
            if (error.name === 'AbortError') {
                _showErrorBanner(
                    'The analysis is taking longer than expected. Please try again.',
                    [],
                    true
                );
            } else {
                _showErrorBanner(
                    'Something went wrong on our end. Please try again in a few seconds.',
                    [],
                    true
                );
            }
        } finally {
            _hideSkeleton();
            _hideProgress();
            setLoading(false);
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!_validatedTicker) return;
        await loadTickerData(_validatedTicker, false);
    });

    timeframeButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            const timeframeKey = String(btn.dataset.timeframe || '').toUpperCase();
            if (!TIMEFRAME_TO_QUERY[timeframeKey]) return;

            setActiveTimeframe(timeframeKey);
            const ticker = currentTicker || input.value.trim().toUpperCase();
            if (!ticker) {
                errorMsg.textContent = 'Enter a ticker first.';
                errorMsg.classList.remove('hidden');
                return;
            }
            currentTicker = ticker;
            await refreshChartForTimeframe(currentTicker, timeframeKey, true);
        });
    });
    setActiveTimeframe(getActiveTimeframe());

    window.addEventListener('resize', () => {
        resizeChartToContainer();
    });

    function openFeedbackModal() {
        if (!feedbackModal) return;
        feedbackModal.classList.remove('hidden');
        if (feedbackMessageEl) {
            feedbackMessageEl.focus();
        }
    }

    function closeFeedbackModal() {
        if (!feedbackModal) return;
        feedbackModal.classList.add('hidden');
    }

    if (feedbackOpenBtn) {
        feedbackOpenBtn.addEventListener('click', () => {
            openFeedbackModal();
        });
    }

    if (feedbackCancelBtn) {
        feedbackCancelBtn.addEventListener('click', () => {
            closeFeedbackModal();
        });
    }

    if (feedbackModal) {
        feedbackModal.addEventListener('click', (event) => {
            if (event.target === feedbackModal) {
                closeFeedbackModal();
            }
        });
    }

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && feedbackModal && !feedbackModal.classList.contains('hidden')) {
            closeFeedbackModal();
        }
    });

    function getFeedbackContext() {
        const tickerFromHeader = String(resTicker?.textContent || '').trim().toUpperCase();
        const tickerFromInput = String(input?.value || '').trim().toUpperCase();
        const ticker = tickerFromHeader || currentTicker || tickerFromInput || '';
        const timeframe = getActiveTimeframe();
        const statusFromLight = String(lightStatusText?.textContent || '').trim();
        const statusFromTrend = String(trendLabelEl?.textContent || '').trim();
        const status = statusFromLight || statusFromTrend || 'UNKNOWN';

        return { ticker, timeframe, status };
    }

    if (feedbackSubmitBtn) {
        feedbackSubmitBtn.addEventListener('click', async () => {
            const message = String(feedbackMessageEl?.value || '').trim();
            if (!message) {
                alert('Please enter feedback before submitting.');
                return;
            }

            const payload = {
                message,
                context: getFeedbackContext(),
            };

            try {
                feedbackSubmitBtn.disabled = true;
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload),
                });
                const body = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(body.error || 'Failed to send feedback.');
                }
                const destination = String(body.saved_to || '').trim();
                alert(destination ? `Feedback sent! Saved to ${destination}` : 'Feedback sent!');
                if (feedbackMessageEl) {
                    feedbackMessageEl.value = '';
                }
                closeFeedbackModal();
            } catch (error) {
                console.error(error);
                alert(error.message || 'Failed to send feedback.');
            } finally {
                feedbackSubmitBtn.disabled = false;
            }
        });
    }

    function setLoading(isLoading) {
        if (isLoading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            timeframeButtons.forEach((btn) => { btn.disabled = true; });
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            // Only re-enable if the current input is still validated
            analyzeBtn.disabled = !_validatedTicker;
            timeframeButtons.forEach((btn) => { btn.disabled = false; });
        }
    }

    function normalizeHistoryPoints(rawHistory) {
        if (!Array.isArray(rawHistory)) return [];
        const normalized = [];
        rawHistory.forEach((point) => {
            if (!point || typeof point !== 'object') return;
            const rawTime = point.time;
            const rawValue = Number(point.value);
            if (!Number.isFinite(rawValue)) return;

            let normalizedTime = null;
            if (typeof rawTime === 'number' && Number.isFinite(rawTime)) {
                const absVal = Math.abs(rawTime);
                if (absVal >= 1e12) {
                    normalizedTime = Math.round(rawTime / 1000); // milliseconds -> seconds
                } else if (absVal >= 1e8) {
                    normalizedTime = Math.round(rawTime); // seconds
                } else {
                    normalizedTime = null; // likely invalid epoch (e.g., 1, 2, 3...)
                }
            } else if (typeof rawTime === 'string') {
                const trimmed = rawTime.trim();
                if (!trimmed) return;
                if (/^\d+$/.test(trimmed)) {
                    const numericTime = Number(trimmed);
                    const absVal = Math.abs(numericTime);
                    if (absVal >= 1e12) {
                        normalizedTime = Math.round(numericTime / 1000);
                    } else if (absVal >= 1e8) {
                        normalizedTime = Math.round(numericTime);
                    } else {
                        normalizedTime = null;
                    }
                } else {
                    normalizedTime = trimmed; // legacy business-day string
                }
            }
            if (normalizedTime === null) return;
            const volume = point.volume !== undefined ? Number(point.volume) : 0;
            const openVal = point.open !== undefined ? Number(point.open) : rawValue;
            const highVal = point.high !== undefined ? Number(point.high) : rawValue;
            const lowVal = point.low !== undefined ? Number(point.low) : rawValue;
            const closeVal = point.close !== undefined ? Number(point.close) : rawValue;
            normalized.push({
                time: normalizedTime,
                value: rawValue,
                volume: volume,
                open: openVal,
                high: highVal,
                low: lowVal,
                close: closeVal
            });
        });

        // Ensure ascending, unique timestamps for Lightweight Charts stability.
        const deduped = new Map();
        normalized.forEach((point) => {
            deduped.set(String(point.time), point);
        });
        return Array.from(deduped.values()).sort((a, b) => {
            const toSortable = (t) => {
                if (typeof t === 'number' && Number.isFinite(t)) return t;
                const parsed = Date.parse(String(t));
                return Number.isFinite(parsed) ? Math.round(parsed / 1000) : 0;
            };
            return toSortable(a.time) - toSortable(b.time);
        });
    }

    function readCrosshairPrice(param, series) {
        if (!param || !series) return null;

        const readFromContainer = (container) => {
            if (!container || typeof container.get !== 'function') return null;
            const entry = container.get(series);
            if (typeof entry === 'number' && Number.isFinite(entry)) return entry;
            if (entry && typeof entry === 'object') {
                if (Number.isFinite(entry.value)) return entry.value;
                if (Number.isFinite(entry.close)) return entry.close;
                if (Number.isFinite(entry.open)) return entry.open;
            }
            return null;
        };

        const fromSeriesData = readFromContainer(param.seriesData);
        if (Number.isFinite(fromSeriesData)) return fromSeriesData;

        const fromSeriesPrices = readFromContainer(param.seriesPrices);
        if (Number.isFinite(fromSeriesPrices)) return fromSeriesPrices;

        return null;
    }

    function formatCrosshairTime(timeValue) {
        let dt = null;
        if (typeof timeValue === 'number' && Number.isFinite(timeValue)) {
            dt = new Date(timeValue * 1000);
        } else if (typeof timeValue === 'string') {
            dt = new Date(timeValue);
        } else if (timeValue && typeof timeValue === 'object') {
            if (
                Number.isFinite(timeValue.year) &&
                Number.isFinite(timeValue.month) &&
                Number.isFinite(timeValue.day)
            ) {
                dt = new Date(Date.UTC(timeValue.year, timeValue.month - 1, timeValue.day));
            }
        }

        if (!(dt instanceof Date) || Number.isNaN(dt.getTime())) {
            return String(timeValue ?? '');
        }
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit', // P2: Detailed timestamps
            timeZoneName: 'short', // P2: Detailed timestamps
            hour12: false,
        }).format(dt);
    }

    function renderChart(history, predictedPrice) {
        if (lwChart) {
            lwChart.remove();
            lwChart = null;
        }
        chartTooltip.style.display = 'none';

        if (!Array.isArray(history) || history.length === 0) {
            if (chartContainer) {
                chartPlaceholder.classList.remove('hidden');
                chartPlaceholder.textContent = "Chart not available.";
            }
            return;
        }

        chartPlaceholder.classList.add('hidden');
        const { width: chartWidth, height: chartHeight } = getChartDimensions();

        const lastValue = Number(history[history.length - 1]?.value);
        const isUptrend = Number.isFinite(predictedPrice) && Number.isFinite(lastValue)
            ? predictedPrice >= lastValue
            : true;
        const lineColor = isUptrend ? '#10b981' : '#ef4444';
        const topColor = isUptrend ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)';
        const bottomColor = isUptrend ? 'rgba(16, 185, 129, 0.0)' : 'rgba(239, 68, 68, 0.0)';

        const cssVars = window.getComputedStyle(document.documentElement);
        const chartTextColor = cssVars.getPropertyValue('--text-muted').trim() || '#9ca3af';
        const chartGridColor = cssVars.getPropertyValue('--chart-border').trim() || 'rgba(255, 255, 255, 0.08)';
        const crosshairColor = cssVars.getPropertyValue('--panel-border').trim() || 'rgba(148, 163, 184, 0.35)';
        const labelBg = cssVars.getPropertyValue('--panel-bg').trim() || 'rgba(15, 23, 42, 0.9)';

        lwChart = LightweightCharts.createChart(chartContainer, {
            width: chartWidth || chartContainer.clientWidth,
            height: chartHeight || 300,
            layout: {
                background: { type: 'solid', color: 'transparent' },
                textColor: chartTextColor,
            },
            grid: {
                vertLines: { color: chartGridColor },
                horzLines: { color: chartGridColor },
            },
            rightPriceScale: {
                visible: true,
                autoScale: true,
                borderVisible: false,
                minimumWidth: 68,
                scaleMargins: { top: 0.08, bottom: 0.08 },
            },
            timeScale: {
                borderVisible: false,
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 2,
                tickMarkFormatter: (time, tickMarkType) => {
                    const d = new Date(time * 1000);
                    const o = { timeZone: 'UTC' };
                    // TickMarkType: 0=Year 1=Month 2=DayOfMonth 3=Time 4=TimeWithSeconds
                    if (tickMarkType === 0)
                        return d.toLocaleDateString('en-US', { ...o, year: 'numeric' });
                    if (tickMarkType === 1)
                        return d.toLocaleDateString('en-US', { ...o, month: 'short' });
                    if (tickMarkType === 2)
                        return d.toLocaleDateString('en-US', { ...o, month: 'short', day: 'numeric' });
                    if (tickMarkType === 3 || tickMarkType === 4)
                        return d.toLocaleTimeString('en-US', { ...o, hour: '2-digit', minute: '2-digit', hour12: false });
                    return d.toLocaleDateString('en-US', o);
                },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {
                    width: 1,
                    color: crosshairColor,
                    style: LightweightCharts.LineStyle.Dashed,
                    labelBackgroundColor: labelBg,
                },
                horzLine: {
                    width: 1,
                    color: crosshairColor,
                    style: LightweightCharts.LineStyle.Dashed,
                    labelBackgroundColor: labelBg,
                },
            },
        });

        let mainSeries;
        if (typeof lwChart.addCandlestickSeries === 'function') {
            mainSeries = lwChart.addCandlestickSeries({
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
                priceFormat: {
                    type: 'price',
                    precision: 2,
                    minMove: 0.01,
                },
            });
        } else {
            // Version 5+ syntax
            mainSeries = lwChart.addSeries(LightweightCharts.CandlestickSeries, {
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
                priceFormat: {
                    type: 'price',
                    precision: 2,
                    minMove: 0.01,
                },
            });
        }

        mainSeries.setData(history);

        let volumeSeries;
        const volumeOptions = {
            color: '#26a69a',
            priceFormat: { type: 'volume' },
            priceScaleId: '', // set as an overlay
        };

        if (typeof lwChart.addHistogramSeries === 'function') {
            volumeSeries = lwChart.addHistogramSeries(volumeOptions);
        } else {
            volumeSeries = lwChart.addSeries(LightweightCharts.HistogramSeries, volumeOptions);
        }

        volumeSeries.priceScale().applyOptions({
            scaleMargins: {
                top: 0.8, // highest point of the series will be at 80% from the top
                bottom: 0,
            },
        });

        const volumeData = history.map((p, index) => {
            let color = '#26a69a'; // green for up
            if (index > 0 && p.value < history[index - 1].value) {
                color = '#ef5350'; // red for down
            }
            return {
                time: p.time,
                value: p.volume || 0,
                color: color
            };
        });
        volumeSeries.setData(volumeData);

        lwChart.subscribeCrosshairMove((param) => {
            if (!param || !param.point || !param.time) {
                chartTooltip.style.display = 'none';
                return;
            }
            if (!chartContainer || param.point.x < 0 || param.point.y < 0) {
                chartTooltip.style.display = 'none';
                return;
            }

            const priceAtCursor = readCrosshairPrice(param, mainSeries);
            if (!Number.isFinite(priceAtCursor)) {
                chartTooltip.style.display = 'none';
                return;
            }
            const timeLabel = formatCrosshairTime(param.time);
            chartTooltip.innerHTML = `<div>${timeLabel}</div><div>${usdFormatter.format(priceAtCursor)}</div>`;
            chartTooltip.style.display = 'block';

            const containerRect = chartContainer.getBoundingClientRect();
            const tooltipWidth = chartTooltip.offsetWidth || 120;
            const tooltipHeight = chartTooltip.offsetHeight || 44;
            const x = Math.min(
                Math.max(6, param.point.x + 12),
                containerRect.width - tooltipWidth - 6
            );
            const y = Math.min(
                Math.max(6, param.point.y + 12),
                containerRect.height - tooltipHeight - 6
            );
            chartTooltip.style.left = `${x}px`;
            chartTooltip.style.top = `${y}px`;
        });

        const lastDataPoint = history[history.length - 1];
        if (Number.isFinite(predictedPrice) && lastDataPoint) {
            const lastTime = lastDataPoint.time;
            let predTime = null;
            if (typeof lastTime === 'number' && Number.isFinite(lastTime)) {
                let stepSeconds = 24 * 60 * 60;
                if (history.length >= 2) {
                    const prevTime = history[history.length - 2].time;
                    if (typeof prevTime === 'number' && Number.isFinite(prevTime) && lastTime > prevTime) {
                        stepSeconds = Math.max(60, Math.round(lastTime - prevTime));
                    }
                }
                predTime = lastTime + stepSeconds;
            } else {
                const predictedDate = new Date(lastTime);
                predictedDate.setDate(predictedDate.getDate() + 1);
                if (predictedDate.getDay() === 6) predictedDate.setDate(predictedDate.getDate() + 2);
                if (predictedDate.getDay() === 0) predictedDate.setDate(predictedDate.getDate() + 1);
                const y = predictedDate.getFullYear();
                const m = String(predictedDate.getMonth() + 1).padStart(2, '0');
                const d = String(predictedDate.getDate()).padStart(2, '0');
                predTime = `${y}-${m}-${d}`;
            }

            let lineSeries;
            const lineOptions = {
                color: '#f59e0b',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
            };
            if (typeof lwChart.addLineSeries === 'function') {
                lineSeries = lwChart.addLineSeries(lineOptions);
            } else {
                lineSeries = lwChart.addSeries(LightweightCharts.LineSeries, lineOptions);
            }
            lineSeries.setData([
                { time: lastDataPoint.time, value: lastDataPoint.value },
                { time: predTime, value: predictedPrice },
            ]);

            if (typeof lineSeries.setMarkers === 'function') {
                lineSeries.setMarkers([
                    {
                        time: predTime,
                        position: 'aboveBar',
                        color: '#f59e0b',
                        shape: 'circle',
                        text: 'Predicted',
                    },
                ]);
            }
        }

        lwChart.timeScale().fitContent();
    }

    function updateDashboard(data) {
        // Meta
        resTicker.textContent = data.meta.symbol;
        const date = new Date(data.meta.generated_at);
        resTime.textContent = `Updated: ${date.toLocaleString()} (${data.meta.mode.toUpperCase()} MODE)`;
        setActiveTimeframe(resolveTimeframeFromMeta(data.meta));

        // Prices
        currentPriceEl.textContent = usdFormatter.format(data.market.current_price);
        predictedPriceEl.textContent = usdFormatter.format(data.market.predicted_price_next_session);

        // Trend
        const trend = data.signals.trend_label.trim();
        trendLabelEl.textContent = trend;
        if (trend.includes('UPTREND')) {
            trendLabelEl.style.color = 'var(--status-green)';
            trendLabelEl.style.border = '1px solid var(--status-green-glow)';
        } else if (trend.includes('DOWNTREND')) {
            trendLabelEl.style.color = 'var(--status-red)';
            trendLabelEl.style.border = '1px solid var(--status-red-glow)';
        } else {
            trendLabelEl.style.color = 'var(--text-main)';
            trendLabelEl.style.border = '1px solid var(--panel-border)';
        }

        // Apply contrarian colour to predicted price:
        // uptrend → red (overbought risk), downtrend → green (opportunity signal)
        predictedPriceEl.classList.remove('price-up', 'price-down');
        if (trend.includes('UPTREND')) {
            predictedPriceEl.classList.add('price-up');
        } else if (trend.includes('DOWNTREND')) {
            predictedPriceEl.classList.add('price-down');
        }

        // Check Engine Light
        engineIndicator.className = 'engine-indicator'; // Reset classes
        const lightString = data.signals.check_engine_light;
        if (lightString.includes('GREEN')) {
            engineIndicator.classList.add('status-green');
            lightStatusText.textContent = "SAFE TO PROCEED";
        } else if (lightString.includes('RED')) {
            engineIndicator.classList.add('status-red');
            lightStatusText.textContent = "RISK DETECTED";
        } else {
            engineIndicator.classList.add('status-yellow');
            lightStatusText.textContent = "NEUTRAL (NOISE)";
        }

        // Sentiment
        const sentiment = data.signals.sentiment_score;
        sentimentScoreEl.textContent = sentiment.toFixed(2);

        sentimentScoreEl.className = '';
        if (sentiment > 0.05) {
            sentimentScoreEl.classList.add('score-positive');
            sentimentDescEl.textContent = 'Positive Sentiment';
        } else if (sentiment < -0.05) {
            sentimentScoreEl.classList.add('score-negative');
            sentimentDescEl.textContent = 'Negative Sentiment';
        } else {
            sentimentScoreEl.classList.add('score-neutral');
            sentimentDescEl.textContent = 'Neutral Sentiment';
        }

        // LLM Insights
        const llmContainer = document.getElementById('llm-insights-container');
        if (llmContainer) {
            llmContainer.innerHTML = '';
            if (data.llm_insights && Object.keys(data.llm_insights).length > 0) {
                for (const [key, report] of Object.entries(data.llm_insights)) {
                    const nameMap = {
                        'chatgpt52': 'ChatGPT 5.2',
                        'deepseek_v3': 'DeepSeek V3',
                        'gemini_v3_pro': 'Gemini V3 Pro'
                    };
                    const modelName = nameMap[key] || key;
                    const div = document.createElement('div');
                    div.className = 'llm-report-item';
                    div.style.padding = '8px';
                    div.style.background = 'rgba(255, 255, 255, 0.05)';
                    div.style.borderRadius = '5px';

                    const llmTrend = String(report?.signals?.trend_label || '').toUpperCase().trim();
                    const llmPrice = Number(report?.market?.predicted_price_next_session);
                    if (!llmPrice && !llmTrend) {
                        console.warn('[IRIS] LLM panel: missing data for model', key, report);
                    }

                    let priceClass = 'llm-price-flat';
                    let trendClass = 'llm-trend-flat';
                    let arrow = '';
                    if (llmTrend.includes('UPTREND')) {
                        priceClass = 'llm-price-up';
                        trendClass = 'llm-trend-up';
                        arrow = '↑ ';
                    } else if (llmTrend.includes('DOWNTREND')) {
                        priceClass = 'llm-price-down';
                        trendClass = 'llm-trend-down';
                        arrow = '↓ ';
                    }

                    div.style.display = 'flex';
                    div.style.justifyContent = 'space-between';
                    div.style.alignItems = 'center';
                    div.innerHTML = `
                        <span style="font-weight:600;font-size:0.95em;">${modelName}</span>
                        <div style="text-align:right;">
                          <div class="${priceClass}">${usdFormatter.format(llmPrice)}</div>
                          <div class="${trendClass}">${arrow}${llmTrend}</div>
                        </div>`;
                    llmContainer.appendChild(div);
                }
            } else {
                llmContainer.innerHTML = '<p class="text-muted">No LLM insights available.</p>';
            }
        }


        // Headlines
        headlinesList.innerHTML = '';
        const headlines = data.evidence.headlines_used;

        if (headlines && headlines.length > 0) {
            headlines.forEach((headline) => {
                const title = typeof headline === 'string'
                    ? headline.trim()
                    : String(headline?.title || headline?.text || '').trim();
                if (!title) return;

                const url         = typeof headline === 'string' ? '' : String(headline?.url || '').trim();
                const publishedAt = typeof headline === 'string' ? '' : String(headline?.published_at || '').trim();
                const dateLabel   = formatHeadlineDate(publishedAt);
                const domain      = extractDomain(url);
                const isLink      = /^https?:\/\//i.test(url);

                const li = document.createElement('li');
                const category = String(typeof headline === 'string' ? 'financial' : (headline?.category || 'financial')).trim().toLowerCase();
                const catClass  = category === 'geopolitical' ? ' headline-item--geo'
                                : category === 'macro'         ? ' headline-item--macro'
                                : '';
                li.className = 'headline-item' + catClass + (isLink ? '' : ' headline-item--no-url');

                // Title — clickable link or plain span
                const titleEl = document.createElement(isLink ? 'a' : 'span');
                titleEl.className = 'headline-title';
                titleEl.textContent = title;
                if (isLink) {
                    titleEl.href = url;
                    titleEl.target = '_blank';
                    titleEl.rel = 'noopener noreferrer';
                }

                // Meta row — date + dot + source domain
                const metaEl = document.createElement('div');
                metaEl.className = 'headline-meta';

                if (dateLabel) {
                    const dateSpan = document.createElement('span');
                    dateSpan.className = 'headline-date';
                    dateSpan.textContent = dateLabel;
                    metaEl.appendChild(dateSpan);
                }

                if (dateLabel && domain) {
                    const dot = document.createElement('span');
                    dot.className = 'headline-dot';
                    metaEl.appendChild(dot);
                }

                if (domain) {
                    const srcSpan = document.createElement('span');
                    srcSpan.className = 'headline-source' + (isLink ? '' : ' headline-source--none');
                    srcSpan.textContent = isLink ? domain : 'no source URL';
                    metaEl.appendChild(srcSpan);
                }

                if (category === 'geopolitical' || category === 'macro') {
                    const tagEl = document.createElement('span');
                    tagEl.className = 'headline-tag';
                    tagEl.textContent = category === 'macro' ? 'Macro' : 'Geopolitical';
                    metaEl.appendChild(tagEl);
                }

                li.appendChild(titleEl);
                if (metaEl.hasChildNodes()) li.appendChild(metaEl);
                headlinesList.appendChild(li);
            });
        }

        // Show scroll hint if list overflows its capped height
        const hintEl = headlinesList.parentElement?.querySelector('.headlines-scroll-hint');
        if (hintEl) {
            const overflows = headlinesList.scrollHeight > headlinesList.clientHeight;
            hintEl.classList.toggle('visible', overflows);
        }

        if (!headlinesList.children.length) {
            const li = document.createElement('li');
            li.className = 'headline-item headline-item--empty';
            li.textContent = 'No recent headlines found.';
            headlinesList.appendChild(li);
        }
    }
});

/* ─────────────────────────────────────────────────────
   Recommended For You – Stock Recommendations
   ───────────────────────────────────────────────────── */

(function initRecommendations() {
  'use strict';

  const recSection   = document.getElementById('recommended-section');
  const recScroll    = document.getElementById('rec-scroll');
  const recSubtitle  = document.getElementById('rec-subtitle');
  const recPrevBtn   = document.getElementById('rec-prev');
  const recNextBtn   = document.getElementById('rec-next');

  if (!recSection || !recScroll) return;

  // ── Sparkline drawing (lightweight canvas, no library) ──
  function drawSparkline(canvas, dataPoints, isPositive) {
    if (!canvas || !dataPoints || dataPoints.length < 2) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    canvas.width  = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width  = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const min = Math.min(...dataPoints);
    const max = Math.max(...dataPoints);
    const range = max - min || 1;
    const pad = 2;

    const color = isPositive ? '#22c55e' : '#ef4444';

    // Area fill
    ctx.beginPath();
    dataPoints.forEach(function (v, i) {
      var x = (i / (dataPoints.length - 1)) * w;
      var y = pad + ((max - v) / range) * (h - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    var grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, isPositive ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line stroke
    ctx.beginPath();
    dataPoints.forEach(function (v, i) {
      var x = (i / (dataPoints.length - 1)) * w;
      var y = pad + ((max - v) / range) * (h - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.stroke();
  }

  // ── Show loading skeleton ──
  function showRecSkeleton() {
    recScroll.innerHTML = '';
    for (var i = 0; i < 5; i++) {
      var skel = document.createElement('div');
      skel.className = 'rec-card-skeleton';
      skel.innerHTML =
        '<div class="rec-skel-line w60"></div>' +
        '<div class="rec-skel-line w40"></div>' +
        '<div class="rec-skel-block"></div>' +
        '<div class="rec-skel-price"></div>';
      recScroll.appendChild(skel);
    }
    recSection.classList.remove('hidden');
  }

  // ── Build one recommendation card ──
  function buildRecCard(item) {
    var pctVal   = item.price_change_pct || 0;
    var isPos    = pctVal > 0;
    var isNeg    = pctVal < 0;
    var dirClass = isPos ? 'rec-positive' : isNeg ? 'rec-negative' : '';
    var badgeCls = isPos ? 'rec-badge-positive' : isNeg ? 'rec-badge-negative' : 'rec-badge-neutral';
    var chCls    = isPos ? 'rec-ch-positive' : isNeg ? 'rec-ch-negative' : 'rec-ch-neutral';
    var sign     = isPos ? '+' : '';

    var card = document.createElement('div');
    card.className = 'rec-card ' + dirClass;
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', 'Analyze ' + item.symbol);

    card.innerHTML =
      '<div class="rec-card-top">' +
        '<span class="rec-ticker">' + item.symbol + '</span>' +
        '<span class="rec-change-badge ' + badgeCls + '">' + sign + pctVal.toFixed(2) + '%</span>' +
      '</div>' +
      '<div class="rec-name">' + (item.name || item.symbol) + '</div>' +
      '<div class="rec-sparkline"><canvas></canvas></div>' +
      '<div class="rec-price-row">' +
        '<span class="rec-price">$' + (item.current_price || 0).toFixed(2) + '</span>' +
        '<span class="rec-price-change ' + chCls + '">' + sign + (item.price_change || 0).toFixed(2) + '</span>' +
      '</div>';

    // Click -> run analysis for this ticker
    function triggerAnalysis() {
      var tickerInput = document.getElementById('ticker-input');
      var form        = document.getElementById('analyze-form');
      if (tickerInput && form) {
        tickerInput.value = item.symbol;
        tickerInput.dispatchEvent(new Event('input', { bubbles: true }));
        setTimeout(function () {
          form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }, 150);
      }
    }
    card.addEventListener('click', triggerAnalysis);
    card.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); triggerAnalysis(); }
    });

    return card;
  }

  // ── Fetch and render recommendations ──
  function fetchRecommendations(ticker) {
    if (!ticker) return;
    showRecSkeleton();
    recSubtitle.textContent = 'Related stocks based on ' + ticker + '\'s sector';

    fetch('/api/related/' + encodeURIComponent(ticker))
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        if (!data.related || data.related.length === 0) {
          recSection.classList.add('hidden');
          return;
        }

        recScroll.innerHTML = '';

        data.related.forEach(function (item) {
          var card = buildRecCard(item);
          recScroll.appendChild(card);

          // Draw sparkline after card is in DOM (needs layout dimensions)
          setTimeout(function () {
            var canvas = card.querySelector('.rec-sparkline canvas');
            if (canvas && item.sparkline && item.sparkline.length >= 2) {
              drawSparkline(canvas, item.sparkline, (item.price_change_pct || 0) >= 0);
            }
          }, 50);
        });

        recSection.classList.remove('hidden');
        updateRecNav();
      })
      .catch(function (err) {
        console.warn('Recommendations fetch failed:', err);
        recSection.classList.add('hidden');
      });
  }

  // ── Scroll navigation ──
  function updateRecNav() {
    if (!recPrevBtn || !recNextBtn) return;
    recPrevBtn.disabled = recScroll.scrollLeft <= 5;
    recNextBtn.disabled = recScroll.scrollLeft + recScroll.clientWidth >= recScroll.scrollWidth - 5;
  }

  if (recPrevBtn) {
    recPrevBtn.addEventListener('click', function () {
      recScroll.scrollBy({ left: -200, behavior: 'smooth' });
      setTimeout(updateRecNav, 400);
    });
  }
  if (recNextBtn) {
    recNextBtn.addEventListener('click', function () {
      recScroll.scrollBy({ left: 200, behavior: 'smooth' });
      setTimeout(updateRecNav, 400);
    });
  }
  recScroll.addEventListener('scroll', updateRecNav);

  // ── Expose globally so the main analysis callback can trigger it ──
  window._irisLoadRecommendations = fetchRecommendations;

})();
