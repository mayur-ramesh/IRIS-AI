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
    // Force en-US locale for all formatting.
    const LOCALE = 'en-US';

    const TIMEFRAME_TO_QUERY = {
        '1D': { period: '1d', interval: '2m' },
        '5D': { period: '5d', interval: '15m' },
        '1M': { period: '1mo', interval: '60m' },
        '6M': { period: '6mo', interval: '1d' },
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
    let currentHorizon = '1D';
    let latestTrajectory = [];
    let latestTrajectoryUpper = [];
    let latestTrajectoryLower = [];
    let latestLlmPredictions = null;

    const HORIZON_LABELS = {
        '1D': '1 Day',
        '5D': '5 Days',
        '1M': '1 Month',
        '6M': '6 Months',
        '1Y': '1 Year',
        '5Y': '5 Years',
    };

    const predictedPriceLabelEl = document.getElementById('predicted-price-label');
    // --- Prediction reasoning tooltip ---
    let activeTooltip = null;
    let activeTooltipTarget = null;

    function showReasoningTooltip(targetEl, modelName, price, signal, reasoning) {
        hideReasoningTooltip();
        const tooltip = document.createElement('div');
        const safeSignal = String(signal || '').trim();
        const signalHtml = safeSignal
            ? `<span class="signal-badge signal-${safeSignal.toLowerCase().replace(/\s+/g, '-')}" style="font-size:0.72em;padding:2px 6px;">${safeSignal}</span>`
            : '';
        tooltip.className = 'prediction-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-model">${modelName}</div>
            <div class="tooltip-details">
                <span>${price || 'N/A'}</span>
                ${signalHtml}
            </div>
            <div class="tooltip-reasoning">${reasoning || 'No reasoning available.'}</div>
        `;

        // Append to dashboard (not the card) to avoid overflow clipping.
        const anchor = document.getElementById('results-dashboard') || document.body;
        anchor.style.position = 'relative';
        anchor.appendChild(tooltip);

        requestAnimationFrame(() => {
            const tRect = targetEl.getBoundingClientRect();
            const aRect = anchor.getBoundingClientRect();
            const tipW = tooltip.offsetWidth;
            const tipH = tooltip.offsetHeight;

            // Try left first.
            let left = tRect.left - aRect.left - tipW - 12;
            let top = tRect.top - aRect.top + (tRect.height / 2) - (tipH / 2);

            // Fallback above.
            if (left < 0) {
                left = tRect.left - aRect.left + (tRect.width / 2) - (tipW / 2);
                top = tRect.top - aRect.top - tipH - 8;
            }

            // Fallback below.
            if (top < 0) {
                left = tRect.left - aRect.left + (tRect.width / 2) - (tipW / 2);
                top = tRect.bottom - aRect.top + 8;
            }

            left = Math.max(4, Math.min(left, aRect.width - tipW - 4));
            top = Math.max(4, top);

            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
            tooltip.classList.add('is-visible');
        });

        activeTooltip = tooltip;
        activeTooltipTarget = targetEl;
    }

    function hideReasoningTooltip() {
        if (activeTooltip) {
            activeTooltip.classList.remove('is-visible');
            const el = activeTooltip;
            setTimeout(() => el.remove(), 200);
            activeTooltip = null;
            activeTooltipTarget = null;
        }
    }
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
            if (btnText) btnText.textContent = `Wait ${remaining}s...`;
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
        if (code) console.debug('[IRIS-AI] Validation rejected 鈥?code:', code, '鈥?ticker:', val);

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
            if (fmtResult.code) console.debug('[IRIS-AI] Format check failed 鈥?code:', fmtResult.code, '鈥?input:', val);
            _setInputState('error');
            _showValidationHint(fmtResult.error, 'error');
            _renderSuggestions([]);
            if (clearBtn) clearBtn.classList.toggle('hidden', !val);
            return;
        }

        // Format OK 鈫?remote check
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

            // Format OK 鈥?debounce the remote call
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
        // Autocomplete on input 鈥?200 ms debounce
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
    const usdFormatter = new Intl.NumberFormat(LOCALE, { style: 'currency', currency: 'USD' });

    const headlineDateFormatter = new Intl.DateTimeFormat(LOCALE, {
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
        d = new Date(num * 1000);   // Unix seconds 鈫?ms
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

    // Chart loading overlay.
    let chartLoadingOverlay = null;
    if (chartContainer) {
        chartLoadingOverlay = document.createElement('div');
        chartLoadingOverlay.className = 'chart-loading-overlay';
        chartLoadingOverlay.innerHTML = `
            <div class="chart-loading-spinner"></div>
            <div class="chart-loading-text">Loading chart data\u2026</div>
        `;
        chartContainer.appendChild(chartLoadingOverlay);
    }

    function showChartLoading(message) {
        if (!chartLoadingOverlay) return;
        const textEl = chartLoadingOverlay.querySelector('.chart-loading-text');
        if (textEl) textEl.textContent = message || 'Loading chart data\u2026';
        chartLoadingOverlay.classList.add('is-active');
    }

    function hideChartLoading() {
        if (chartLoadingOverlay) {
            chartLoadingOverlay.classList.remove('is-active');
        }
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
            renderChart(
                history,
                latestPredictedPrice,
                latestTrajectory,
                latestTrajectoryUpper,
                latestTrajectoryLower,
            );
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
                renderChart(
                    fallback.history,
                    latestPredictedPrice,
                    latestTrajectory,
                    latestTrajectoryUpper,
                    latestTrajectoryLower,
                );
                errorMsg.classList.add('hidden');
                return;
            } catch (fallbackError) {
                if (requestId !== historyRequestId) return;
                console.error(fallbackError);
            }

            if (Array.isArray(latestAnalyzeHistory) && latestAnalyzeHistory.length > 0) {
                renderChart(
                    latestAnalyzeHistory,
                    latestPredictedPrice,
                    latestTrajectory,
                    latestTrajectoryUpper,
                    latestTrajectoryLower,
                );
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

    function applyTrendBadge(trend) {
        const normalizedTrend = String(trend || '').trim();
        trendLabelEl.textContent = normalizedTrend || 'UNKNOWN';
        if (normalizedTrend.includes('UPTREND')) {
            trendLabelEl.style.color = 'var(--status-green)';
            trendLabelEl.style.border = '1px solid var(--status-green-glow)';
        } else if (normalizedTrend.includes('DOWNTREND')) {
            trendLabelEl.style.color = 'var(--status-red)';
            trendLabelEl.style.border = '1px solid var(--status-red-glow)';
        } else {
            trendLabelEl.style.color = 'var(--text-main)';
            trendLabelEl.style.border = '1px solid var(--panel-border)';
        }
    }

    function renderInvestmentSignalBadge(signalValue) {
        const signalStr = String(signalValue || '').trim();
        let existingSignalBadge = document.getElementById('investment-signal-badge');
        if (!existingSignalBadge && trendLabelEl?.parentElement) {
            existingSignalBadge = document.createElement('div');
            existingSignalBadge.id = 'investment-signal-badge';
            trendLabelEl.parentElement.appendChild(existingSignalBadge);
        }
        if (!existingSignalBadge) return;
        if (signalStr) {
            const signalClass = 'signal-' + signalStr.toLowerCase().replace(/\s+/g, '-');
            existingSignalBadge.className = 'signal-badge ' + signalClass;
            existingSignalBadge.textContent = signalStr;
        } else {
            existingSignalBadge.className = 'signal-badge signal-hold';
            existingSignalBadge.textContent = 'HOLD';
        }
    }

    function renderLlmInsights(llmData) {
        const llmContainer = document.getElementById('llm-insights-container');
        if (!llmContainer) return;

        llmContainer.innerHTML = '';
        if (!llmData || Object.keys(llmData).length === 0) {
            llmContainer.innerHTML = '<p class="text-muted">No LLM insights available.</p>';
            return;
        }

        for (const [key, report] of Object.entries(llmData)) {
            const nameMap = {
                chatgpt52: 'ChatGPT 5.2',
                deepseek_v3: 'DeepSeek V3',
                gemini_v3_pro: 'Gemini V3 Pro',
            };
            const modelName = nameMap[key] || key;
            const div = document.createElement('div');
            div.className = 'llm-report-item prediction-result';
            div.style.padding = '8px';
            div.style.background = 'rgba(255, 255, 255, 0.05)';
            div.style.borderRadius = '5px';
            div.style.display = 'flex';
            div.style.justifyContent = 'space-between';
            div.style.alignItems = 'center';

            const llmTrend = String(report?.signals?.trend_label || '').toUpperCase().trim();
            // Strip non-ASCII characters (e.g., emoji that can render as garbled symbols on CJK systems).
            const cleanTrend = llmTrend.replace(/[^\x20-\x7E]/g, '').trim();
            const llmSignal = String(report?.signals?.investment_signal || '').trim();
            const llmPrice = Number(
                report?.market?.predicted_price_horizon ?? report?.market?.predicted_price_next_session
            );
            const reasoning = String(report?.reasoning || '').trim();

            let priceClass = 'llm-price-flat';
            let trendClass = 'llm-trend-flat';
            let arrow = '';
            if (cleanTrend.includes('UPTREND')) {
                priceClass = 'llm-price-up';
                trendClass = 'llm-trend-up';
                arrow = '\u2191 ';
            } else if (cleanTrend.includes('DOWNTREND')) {
                priceClass = 'llm-price-down';
                trendClass = 'llm-trend-down';
                arrow = '\u2193 ';
            }

            let signalHtml = '';
            if (llmSignal) {
                const sc = 'signal-' + llmSignal.toLowerCase().replace(/\s+/g, '-');
                signalHtml = `<div class="signal-badge ${sc}" style="font-size:0.7em;padding:2px 8px;margin-top:4px;">${llmSignal}</div>`;
            }

            div.innerHTML = `
                <span style="font-weight:600;font-size:0.95em;">${modelName}</span>
                <div style="text-align:right;">
                  <div class="${priceClass}">${isFinite(llmPrice) ? usdFormatter.format(llmPrice) : 'N/A'}</div>
                  <div class="${trendClass}">${arrow}${cleanTrend || 'N/A'}</div>
                  ${signalHtml}
                </div>`;

            div.addEventListener('mouseenter', () => {
                const priceStr = isFinite(llmPrice) ? usdFormatter.format(llmPrice) : 'N/A';
                showReasoningTooltip(div, modelName, priceStr, llmSignal, reasoning);
            });
            div.addEventListener('mouseleave', hideReasoningTooltip);
            llmContainer.appendChild(div);
        }
    }

    async function refreshLlmPredictions(ticker, horizon) {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        const normalizedHorizon = String(horizon || '').trim().toUpperCase() || '1D';
        if (!normalizedTicker) return;
        try {
            const resp = await fetch(
                `/api/llm-predict?ticker=${encodeURIComponent(normalizedTicker)}&horizon=${encodeURIComponent(normalizedHorizon)}`
            );
            if (!resp.ok) return;
            const body = await resp.json();
            latestLlmPredictions = body?.models || null;
            if (latestLlmPredictions) {
                renderLlmInsights(latestLlmPredictions);
            }
        } catch (err) {
            console.warn('LLM prediction update failed:', err);
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
        params.set('horizon', currentHorizon);

        _retryTicker = normalizedTicker;

        // 30-second hard timeout on the entire analysis
        const timeoutCtrl = new AbortController();
        const timeoutId = setTimeout(() => timeoutCtrl.abort(), 30000);

        setLoading(true);
        _hideErrorBanner();
        errorMsg.classList.add('hidden');
        _showProgress();
        _showSkeleton();
        showChartLoading('Analyzing ticker\u2026');
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
            latestLlmPredictions = null;
            latestPredictedPrice = Number(data?.market?.predicted_price_horizon ?? data?.market?.predicted_price_next_session);
            latestTrajectory = Array.isArray(data?.market?.prediction_trajectory) ? data.market.prediction_trajectory : [];
            latestTrajectoryUpper = Array.isArray(data?.market?.prediction_trajectory_upper) ? data.market.prediction_trajectory_upper : [];
            latestTrajectoryLower = Array.isArray(data?.market?.prediction_trajectory_lower) ? data.market.prediction_trajectory_lower : [];
            latestAnalyzeHistory = normalizeHistoryPoints(data?.market?.history);
            latestAnalyzeTimeframe = getActiveTimeframe();
            try { updateDashboard(data); } catch (renderErr) { console.error('[IRIS] updateDashboard error:', renderErr); }
            await refreshChartForTimeframe(currentTicker, getActiveTimeframe(), false);
            await refreshLlmPredictions(currentTicker, currentHorizon);
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
            hideChartLoading();
            setLoading(false);
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!_validatedTicker) return;
        await loadTickerData(_validatedTicker, false);
    });

    // Expose analysis function so recommendation cards can invoke it directly
    window._irisAnalyzeTicker = function (ticker) {
        const sym = String(ticker || '').trim().toUpperCase();
        if (!sym) return;
        _validatedTicker = sym;
        loadTickerData(sym, false);
    };

    function setActiveHorizon(horizonKey) {
        const normalized = String(horizonKey || '').trim().toUpperCase();
        currentHorizon = HORIZON_LABELS[normalized] ? normalized : '1D';
        const label = HORIZON_LABELS[currentHorizon] || '1 Day';
        if (predictedPriceLabelEl) {
            predictedPriceLabelEl.textContent = `Predicted (${label})`;
        }
    }

    timeframeButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            const timeframeKey = String(btn.dataset.timeframe || '').toUpperCase();
            if (!TIMEFRAME_TO_QUERY[timeframeKey]) return;

            const ticker = currentTicker || input.value.trim().toUpperCase();
            if (!ticker) {
                errorMsg.textContent = 'Enter a ticker first.';
                errorMsg.classList.remove('hidden');
                return;
            }
            currentTicker = ticker;

            // Show loading states.
            setActiveTimeframe(timeframeKey);
            btn.classList.add('is-loading');
            showChartLoading('Updating chart\u2026');
            timeframeButtons.forEach((b) => { b.disabled = true; });

            const priceCard = document.querySelector('.price-card');
            try {
                // 1) Refresh chart history (lightweight)
                await refreshChartForTimeframe(currentTicker, timeframeKey, false);

                // 2) If horizon changed, run prediction
                const newHorizon = timeframeKey;
                if (HORIZON_LABELS[newHorizon]) {
                    const horizonChanged = newHorizon !== currentHorizon;
                    setActiveHorizon(newHorizon);
                    if (horizonChanged) {
                        showChartLoading('Running prediction model\u2026');
                        if (priceCard) priceCard.classList.add('prediction-updating');
                        try {
                            const predResp = await fetch(
                                `/api/predict?ticker=${encodeURIComponent(ticker)}&horizon=${encodeURIComponent(newHorizon)}`
                            );
                            if (predResp.ok) {
                                const pred = await predResp.json();
                                latestPredictedPrice = Number(pred?.predicted_price);
                                latestTrajectory = Array.isArray(pred?.prediction_trajectory) ? pred.prediction_trajectory : [];
                                latestTrajectoryUpper = Array.isArray(pred?.prediction_trajectory_upper) ? pred.prediction_trajectory_upper : [];
                                latestTrajectoryLower = Array.isArray(pred?.prediction_trajectory_lower) ? pred.prediction_trajectory_lower : [];

                                if (predictedPriceEl && Number.isFinite(latestPredictedPrice)) {
                                    predictedPriceEl.textContent = usdFormatter.format(latestPredictedPrice);
                                }

                                const trend = String(pred?.trend_label || '').replace(/[^\x20-\x7E]/g, '').trim();
                                applyTrendBadge(trend);
                                renderInvestmentSignalBadge(pred?.investment_signal || '');
                                renderChart(
                                    latestAnalyzeHistory,
                                    latestPredictedPrice,
                                    latestTrajectory,
                                    latestTrajectoryUpper,
                                    latestTrajectoryLower,
                                );
                                await refreshLlmPredictions(currentTicker, currentHorizon);
                            }
                        } catch (err) {
                            console.warn('Prediction update failed:', err);
                        } finally {
                            if (priceCard) priceCard.classList.remove('prediction-updating');
                        }
                    }
                }
            } catch (error) {
                console.error('Timeframe switch error:', error);
            } finally {
                btn.classList.remove('is-loading');
                hideChartLoading();
                if (priceCard) priceCard.classList.remove('prediction-updating');
                timeframeButtons.forEach((b) => { b.disabled = false; });
            }
        });
    });
    setActiveTimeframe(getActiveTimeframe());
    setActiveHorizon(currentHorizon);

    // Mobile: tap to toggle prediction tooltips.
    document.addEventListener('touchstart', (e) => {
        const target = e.target.closest('.prediction-result');
        if (!target) {
            hideReasoningTooltip();
            return;
        }
        if (activeTooltip && activeTooltipTarget === target) {
            hideReasoningTooltip();
        }
    });

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
        return new Intl.DateTimeFormat(LOCALE, {
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

    function renderChart(history, predictedPrice, trajectory, trajectoryUpper = [], trajectoryLower = []) {
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
                scaleMargins: { top: 0.05, bottom: 0.15 },
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
                        return d.toLocaleDateString(LOCALE, { ...o, year: 'numeric' });
                    if (tickMarkType === 1)
                        return d.toLocaleDateString(LOCALE, { ...o, month: 'short' });
                    if (tickMarkType === 2)
                        return d.toLocaleDateString(LOCALE, { ...o, month: 'short', day: 'numeric' });
                    if (tickMarkType === 3 || tickMarkType === 4)
                        return d.toLocaleTimeString(LOCALE, { ...o, hour: '2-digit', minute: '2-digit', hour12: false });
                    return d.toLocaleDateString(LOCALE, o);
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
            priceScaleId: 'volume_scale',
        };

        if (typeof lwChart.addHistogramSeries === 'function') {
            volumeSeries = lwChart.addHistogramSeries(volumeOptions);
        } else {
            volumeSeries = lwChart.addSeries(LightweightCharts.HistogramSeries, volumeOptions);
        }

        volumeSeries.priceScale().applyOptions({
            scaleMargins: {
                top: 0.82,
                bottom: 0,
            },
            drawTicks: false,
            borderVisible: false,
            visible: false,
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
        if (volumeData.some((d) => d.value > 0)) {
            volumeSeries.setData(volumeData);
        }

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
            const isUpForecast = predictedPrice >= lastDataPoint.value;
            const forecastColor = isUpForecast ? '#06b6d4' : '#f97316';

            const trajPoints = Array.isArray(trajectory) && trajectory.length > 0 ? trajectory : [predictedPrice];
            const forecastData = [];
            const upperPoints = Array.isArray(trajectoryUpper) ? trajectoryUpper : [];
            const lowerPoints = Array.isArray(trajectoryLower) ? trajectoryLower : [];
            const upperData = [];
            const lowerData = [];

            let stepSeconds = 24 * 60 * 60;
            if (typeof lastTime === 'number' && Number.isFinite(lastTime)) {
                if (history.length >= 2) {
                    const prevTime = history[history.length - 2].time;
                    if (typeof prevTime === 'number' && Number.isFinite(prevTime) && lastTime > prevTime) {
                        stepSeconds = Math.max(60, Math.round(lastTime - prevTime));
                    }
                }
            }

            forecastData.push({ time: lastDataPoint.time, value: lastDataPoint.value });
            upperData.push({ time: lastDataPoint.time, value: Number(lastDataPoint.value) });
            lowerData.push({ time: lastDataPoint.time, value: Number(lastDataPoint.value) });

            const HORIZON_DAYS = {
                '1D': 1, '5D': 5, '1M': 21, '6M': 126, '1Y': 252, '5Y': 1260,
            };
            const totalDays = HORIZON_DAYS[currentHorizon] || 1;

            for (let i = 0; i < trajPoints.length; i++) {
                const dayOffset = trajPoints.length === 1
                    ? totalDays
                    : Math.round(((i + 1) / trajPoints.length) * totalDays);

                if (typeof lastTime === 'number' && Number.isFinite(lastTime)) {
                    const timeValue = lastTime + (dayOffset * stepSeconds);
                    forecastData.push({
                        time: timeValue,
                        value: trajPoints[i],
                    });
                    const fallbackUpper = Number(trajPoints[i]) * 1.02;
                    const fallbackLower = Number(trajPoints[i]) * 0.98;
                    upperData.push({
                        time: timeValue,
                        value: Number.isFinite(Number(upperPoints[i])) ? Number(upperPoints[i]) : fallbackUpper,
                    });
                    lowerData.push({
                        time: timeValue,
                        value: Number.isFinite(Number(lowerPoints[i])) ? Number(lowerPoints[i]) : fallbackLower,
                    });
                } else {
                    const d = new Date(lastTime);
                    let added = 0;
                    while (added < dayOffset) {
                        d.setDate(d.getDate() + 1);
                        if (d.getDay() !== 0 && d.getDay() !== 6) added++;
                    }
                    const y = d.getFullYear();
                    const m = String(d.getMonth() + 1).padStart(2, '0');
                    const dd = String(d.getDate()).padStart(2, '0');
                    const timeValue = `${y}-${m}-${dd}`;
                    forecastData.push({ time: timeValue, value: trajPoints[i] });
                    const fallbackUpper = Number(trajPoints[i]) * 1.02;
                    const fallbackLower = Number(trajPoints[i]) * 0.98;
                    upperData.push({
                        time: timeValue,
                        value: Number.isFinite(Number(upperPoints[i])) ? Number(upperPoints[i]) : fallbackUpper,
                    });
                    lowerData.push({
                        time: timeValue,
                        value: Number.isFinite(Number(lowerPoints[i])) ? Number(lowerPoints[i]) : fallbackLower,
                    });
                }
            }

            let forecastSeries;
            const forecastOpts = {
                color: forecastColor,
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.LargeDashed,
                crosshairMarkerVisible: true,
                crosshairMarkerRadius: 5,
                priceLineVisible: false,
                lastValueVisible: true,
            };
            if (typeof lwChart.addLineSeries === 'function') {
                forecastSeries = lwChart.addLineSeries(forecastOpts);
            } else {
                forecastSeries = lwChart.addSeries(LightweightCharts.LineSeries, forecastOpts);
            }
            forecastSeries.setData(forecastData);

            const upperBandOpts = {
                color: isUpForecast ? 'rgba(6, 182, 212, 0.25)' : 'rgba(249, 115, 22, 0.25)',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
            };
            const lowerBandOpts = {
                color: isUpForecast ? 'rgba(6, 182, 212, 0.25)' : 'rgba(249, 115, 22, 0.25)',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
            };
            let upperBandSeries;
            let lowerBandSeries;
            if (typeof lwChart.addLineSeries === 'function') {
                upperBandSeries = lwChart.addLineSeries(upperBandOpts);
                lowerBandSeries = lwChart.addLineSeries(lowerBandOpts);
            } else {
                upperBandSeries = lwChart.addSeries(LightweightCharts.LineSeries, upperBandOpts);
                lowerBandSeries = lwChart.addSeries(LightweightCharts.LineSeries, lowerBandOpts);
            }
            upperBandSeries.setData(upperData);
            lowerBandSeries.setData(lowerData);

            const lastForecast = forecastData[forecastData.length - 1];
            if (lastForecast && typeof forecastSeries.setMarkers === 'function') {
                const horizonText = HORIZON_LABELS[currentHorizon] || 'Predicted';
                forecastSeries.setMarkers([
                    {
                        time: lastForecast.time,
                        position: isUpForecast ? 'aboveBar' : 'belowBar',
                        color: forecastColor,
                        shape: 'circle',
                        text: `${horizonText}: $${predictedPrice.toFixed(2)}`,
                    },
                ]);
            }
        }
        // Extend right offset to give space for the forecast
        const HORIZON_OFFSET = {
            '1D': 2, '5D': 4, '1M': 6, '6M': 10, '1Y': 12, '5Y': 14,
        };
        lwChart.applyOptions({
            timeScale: { rightOffset: HORIZON_OFFSET[currentHorizon] || 2 },
        });

        lwChart.timeScale().fitContent();
    }

    function updateDashboard(data) {
        // Meta
        resTicker.textContent = data?.meta?.symbol || '??';
        const date = new Date(data?.meta?.generated_at);
        const modeStr = (data?.meta?.mode || 'live').toUpperCase();
        resTime.textContent = `Updated: ${date.toLocaleString(LOCALE)} (${modeStr} MODE)`;
        setActiveTimeframe(resolveTimeframeFromMeta(data?.meta || {}));

        // Sync horizon state from response
        const respHorizon = data?.meta?.risk_horizon || currentHorizon;
        setActiveHorizon(respHorizon);

        // Prices
        const currentPrice = Number(data?.market?.current_price);
        const predictedPrice = Number(data?.market?.predicted_price_horizon ?? data?.market?.predicted_price_next_session);
        currentPriceEl.textContent = isFinite(currentPrice) ? usdFormatter.format(currentPrice) : 'N/A';
        predictedPriceEl.textContent = isFinite(predictedPrice) ? usdFormatter.format(predictedPrice) : 'N/A';

        // Trend
        const trend = (data?.signals?.trend_label || '').replace(/[^\x20-\x7E]/g, '').trim();
        applyTrendBadge(trend);

        // Apply contrarian colour to predicted price:
        // uptrend 鈫?red (overbought risk), downtrend 鈫?green (opportunity signal)
        predictedPriceEl.classList.remove('price-up', 'price-down');
        if (trend.includes('UPTREND')) {
            predictedPriceEl.classList.add('price-up');
        } else if (trend.includes('DOWNTREND')) {
            predictedPriceEl.classList.add('price-down');
        }

        // Investment signal badge
        renderInvestmentSignalBadge(data?.signals?.investment_signal || '');

        // Make IRIS prediction card hoverable for reasoning.
        const priceCard = document.querySelector('.price-card');
        if (priceCard) {
            const irisReasoning = data?.signals?.iris_reasoning?.summary || '';
            priceCard.classList.add('prediction-result');
            priceCard.onmouseenter = () => {
                const priceStr = predictedPriceEl.textContent;
                const signal = data?.signals?.investment_signal || '';
                showReasoningTooltip(priceCard, 'IRIS Model', priceStr, signal, irisReasoning);
            };
            priceCard.onmouseleave = hideReasoningTooltip;
        }

        // Check Engine Light
        engineIndicator.className = 'engine-indicator'; // Reset classes
        const lightString = data?.signals?.check_engine_light || '';
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
        const sentiment = Number(data?.signals?.sentiment_score ?? 0);
        sentimentScoreEl.textContent = isFinite(sentiment) ? sentiment.toFixed(2) : '0.00';

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

        // LLM Insights (prefer live /api/llm-predict results when available)
        const llmData = latestLlmPredictions && Object.keys(latestLlmPredictions).length
            ? latestLlmPredictions
            : data.llm_insights;
        renderLlmInsights(llmData);


        // Headlines
        headlinesList.innerHTML = '';
        const headlines = data?.evidence?.headlines_used || [];

        if (headlines && headlines.length > 0) {
            headlines.forEach((headline) => {
                const title = typeof headline === 'string'
                    ? headline.trim()
                    : String(headline?.title || headline?.text || '').trim();
                if (!title) return;

                // Enforce: only display headlines with a valid clickable URL
                const url = typeof headline === 'string' ? '' : String(headline?.url || '').trim();
                if (!url || (!url.startsWith('http://') && !url.startsWith('https://'))) return;

                const publishedAt = typeof headline === 'string' ? '' : String(headline?.published_at || '').trim();
                const dateLabel   = formatHeadlineDate(publishedAt);
                const domain      = extractDomain(url);

                const li = document.createElement('li');
                const category = String(typeof headline === 'string' ? 'financial' : (headline?.category || 'financial')).trim().toLowerCase();
                const catClass  = category === 'geopolitical' ? ' headline-item--geo'
                                : category === 'macro'         ? ' headline-item--macro'
                                : '';
                li.className = 'headline-item' + catClass;

                // Title 鈥?always a clickable link
                const titleEl = document.createElement('a');
                titleEl.className = 'headline-title';
                titleEl.textContent = title;
                titleEl.href = url;
                titleEl.target = '_blank';
                titleEl.rel = 'noopener noreferrer';

                // Meta row 鈥?date + dot + source domain
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
                    srcSpan.className = 'headline-source';
                    srcSpan.textContent = domain;
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
            li.textContent = 'No linked headlines available for this ticker.';
            headlinesList.appendChild(li);
        }
    }
});

/* 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
   Recommended For You 鈥?Stock Recommendations
   鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ */

(function initRecommendations() {
  'use strict';

  const recSection   = document.getElementById('recommended-section');
  const recScroll    = document.getElementById('rec-scroll');
  const recSubtitle  = document.getElementById('rec-subtitle');
  const recPrevBtn   = document.getElementById('rec-prev');
  const recNextBtn   = document.getElementById('rec-next');

  if (!recSection || !recScroll) return;

  // 鈹€鈹€ Sparkline drawing (lightweight canvas, no library) 鈹€鈹€
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

  // 鈹€鈹€ Show loading skeleton 鈹€鈹€
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

  // 鈹€鈹€ Build one recommendation card 鈹€鈹€
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
      if (tickerInput) tickerInput.value = item.symbol;
      window.scrollTo({ top: 0, behavior: 'smooth' });
      if (typeof window._irisAnalyzeTicker === 'function') {
        window._irisAnalyzeTicker(item.symbol);
      }
    }
    card.addEventListener('click', triggerAnalysis);
    card.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); triggerAnalysis(); }
    });

    return card;
  }

  // 鈹€鈹€ Fetch and render recommendations 鈹€鈹€
  function fetchRecommendations(ticker) {
    if (!ticker) return;
    showRecSkeleton();
    if (recSubtitle) recSubtitle.textContent = 'Related stocks based on ' + ticker + '\'s sector';

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

  // 鈹€鈹€ Scroll navigation 鈹€鈹€
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

  // 鈹€鈹€ Expose globally so the main analysis callback can trigger it 鈹€鈹€
  window._irisLoadRecommendations = fetchRecommendations;

})();

