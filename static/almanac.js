(function () {
    'use strict';

    const PRIMARY_THEME_KEY = 'iris-theme';
    const SECONDARY_THEME_KEY = 'iris-theme-preference';
    const SPY_TICKER = 'SPY';
    const DEFAULT_DATE = '2026-04-09';

    let currentView = 'daily';
    let currentDate = DEFAULT_DATE;
    let currentWeekStart = '2026-04-06';
    let currentMonth = '2026-04';
    let almanacData = null;
    let almanacPromise = null;
    let accuracyCache = null;
    let accuracyPromise = null;
    let weeklyAccuracyCache = {};
    let weeklyAccuracyPromiseCache = {};
    let monthDetailCache = {};
    let monthDetailPromiseCache = {};
    let irisLiveData = null;
    let irisPromise = null;
    let tradingDates = [];
    let monthKeys = [];
    let seasonalTooltip = null;
    let monthlyTooltip = null;
    let activeTooltipTarget = null;
    let activeMonthlyTooltipTarget = null;
    let currentMonthCalendarLookup = {};

    const tabs = Array.from(document.querySelectorAll('.alm-tab'));
    const views = Array.from(document.querySelectorAll('.alm-view'));
    const themeToggle = document.getElementById('theme-toggle');
    const SEASONAL_MONTH_ORDER = [
        '2026-11',
        '2026-12',
        '2026-01',
        '2026-02',
        '2026-03',
        '2026-04',
        '2026-05',
        '2026-06',
        '2026-07',
        '2026-08',
        '2026-09',
        '2026-10',
    ];
    const SEASONAL_PHASES = {
        bullish: new Set(['2026-11', '2026-12', '2026-01', '2026-02', '2026-03', '2026-04']),
        bearish: new Set(['2026-05', '2026-06', '2026-07', '2026-08', '2026-09', '2026-10']),
    };
    const SEASONAL_SIGNAL_META = {
        timing: {
            className: 'sg-g',
            category: 'Timing edge',
            meaning: 'Recurring seasonal setup with a historically constructive timing bias.',
            order: 1,
        },
        weekly_timing: {
            className: 'sg-a',
            category: 'Weekly pattern',
            meaning: 'Shorter seasonal window or rally pattern that tends to matter over a few sessions or weeks.',
            order: 2,
        },
        macro_context: {
            className: 'sg-b',
            category: 'Macro context',
            meaning: 'Broader election-cycle or macro seasonal backdrop that frames the month rather than a single entry signal.',
            order: 3,
        },
        risk_indicator: {
            className: 'sg-r',
            category: 'Risk indicator',
            meaning: 'Higher-stakes warning or historical indicator worth watching for downside or regime risk.',
            order: 4,
        },
        default: {
            className: 'sg-n',
            category: 'Reference',
            meaning: 'General seasonal context from the Almanac source.',
            order: 9,
        },
    };
    const WEEKLY_INDEX_META = [
        { scoreKey: 'd', directionKey: 'd_dir', accuracyKey: 'd', pctKey: 'dji', weeklyKey: 'dow', label: 'Dow', shortLabel: 'Dow' },
        { scoreKey: 's', directionKey: 's_dir', accuracyKey: 's', pctKey: 'sp500', weeklyKey: 'sp500', label: 'S&P 500', shortLabel: 'S&P' },
        { scoreKey: 'n', directionKey: 'n_dir', accuracyKey: 'n', pctKey: 'nasdaq', weeklyKey: 'nasdaq', label: 'NASDAQ', shortLabel: 'Nas' },
    ];

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function readStoredTheme() {
        try {
            const primary = localStorage.getItem(PRIMARY_THEME_KEY);
            if (primary === 'dark' || primary === 'light') {
                return primary;
            }
            const secondary = localStorage.getItem(SECONDARY_THEME_KEY);
            if (secondary === 'dark' || secondary === 'light') {
                return secondary;
            }
        } catch (error) {
            return null;
        }
        return null;
    }

    function writeStoredTheme(theme) {
        try {
            localStorage.setItem(PRIMARY_THEME_KEY, theme);
            localStorage.setItem(SECONDARY_THEME_KEY, theme);
        } catch (error) {
            // Ignore storage failures.
        }
    }

    function applyTheme(theme) {
        const normalized = theme === 'dark' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', normalized);
        if (themeToggle) {
            themeToggle.setAttribute('aria-pressed', normalized === 'dark' ? 'true' : 'false');
        }
    }

    function initializeTheme() {
        const stored = readStoredTheme();
        if (stored) {
            applyTheme(stored);
            return;
        }
        const preferDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        applyTheme(preferDark ? 'dark' : 'light');
    }

    async function fetchJson(url) {
        const response = await fetch(url, {
            headers: {
                Accept: 'application/json',
            },
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.error || 'Request failed');
        }
        return payload;
    }

    async function loadAlmanacFull() {
        if (almanacData) {
            return almanacData;
        }
        if (almanacPromise) {
            return almanacPromise;
        }
        almanacPromise = Promise.all([
            fetchJson('/api/almanac/daily'),
            fetchJson('/api/almanac/seasonal'),
        ])
            .then(([dailyPayload, seasonalPayload]) => {
                almanacData = {
                    daily: dailyPayload.daily || {},
                    heatmap: seasonalPayload.heatmap || {},
                    signals: seasonalPayload.signals || [],
                    months: seasonalPayload.months || {},
                };
                tradingDates = Object.keys(almanacData.daily).sort();
                monthKeys = Object.keys(almanacData.months).sort();
                return almanacData;
            })
            .catch((error) => {
                console.error('Failed to load almanac data:', error);
                almanacPromise = null;
                return null;
            });
        return almanacPromise;
    }

    async function loadAccuracyData() {
        if (accuracyCache) return accuracyCache;
        if (accuracyPromise) return accuracyPromise;
        accuracyPromise = fetchJson('/api/almanac/accuracy')
            .then((payload) => {
                if (payload.available === false) return null;
                accuracyCache = payload;
                return payload;
            })
            .catch(() => {
                accuracyPromise = null;
                return null;
            });
        return accuracyPromise;
    }

    async function loadWeekAccuracy(startDate) {
        const key = String(startDate || '').trim();
        if (!key) {
            return null;
        }
        if (weeklyAccuracyCache[key]) {
            return weeklyAccuracyCache[key];
        }
        if (weeklyAccuracyPromiseCache[key]) {
            return weeklyAccuracyPromiseCache[key];
        }
        weeklyAccuracyPromiseCache[key] = fetchJson('/api/almanac/accuracy/week?start=' + encodeURIComponent(key))
            .then((payload) => {
                weeklyAccuracyCache[key] = payload;
                delete weeklyAccuracyPromiseCache[key];
                return payload;
            })
            .catch(() => {
                delete weeklyAccuracyPromiseCache[key];
                return null;
            });
        return weeklyAccuracyPromiseCache[key];
    }

    async function loadMonthDetail(monthKey) {
        const key = String(monthKey || '').trim();
        if (!key) {
            return null;
        }
        if (monthDetailCache[key]) {
            return monthDetailCache[key];
        }
        if (monthDetailPromiseCache[key]) {
            return monthDetailPromiseCache[key];
        }
        monthDetailPromiseCache[key] = fetchJson('/api/almanac/month/' + encodeURIComponent(key))
            .then((payload) => {
                monthDetailCache[key] = payload;
                delete monthDetailPromiseCache[key];
                return payload;
            })
            .catch((error) => {
                console.error('Failed to load month detail:', error);
                delete monthDetailPromiseCache[key];
                return null;
            });
        return monthDetailPromiseCache[key];
    }

    async function fetchIrisLive() {
        if (irisLiveData) {
            return irisLiveData;
        }
        if (irisPromise) {
            return irisPromise;
        }
        irisPromise = fetchJson('/api/almanac/iris-snapshot')
            .then((payload) => {
                irisLiveData = payload;
                return payload;
            })
            .catch((error) => {
                console.error('IRIS snapshot failed:', error);
                irisPromise = null;
                return null;
            });
        return irisPromise;
    }

    function scoreClass(score) {
        const numeric = Number(score);
        if (numeric >= 60) {
            return 'alm-score-bullish';
        }
        if (numeric <= 40) {
            return 'alm-score-bearish';
        }
        return 'alm-score-sideways';
    }

    function directionBadge(direction) {
        const normalized = String(direction || '').toUpperCase();
        const label = normalized === 'D' ? 'Bullish' : normalized === 'N' ? 'Bearish' : 'Sideways';
        const className = normalized === 'D' ? 'alm-dir-d' : normalized === 'N' ? 'alm-dir-n' : 'alm-dir-s';
        return '<span class="alm-direction-badge ' + className + '">' + label + '</span>';
    }

    function signalBadge(signal) {
        const normalized = String(signal || '').trim().toUpperCase();
        const supported = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'];
        if (!supported.includes(normalized)) {
            return '<span class="alm-chip">' + escapeHtml(normalized || 'N/A') + '</span>';
        }
        const className = 'signal-' + normalized.toLowerCase().replace(/\s+/g, '-');
        return '<span class="signal-badge ' + className + '">' + escapeHtml(normalized) + '</span>';
    }

    function agreementBadge(result) {
        const className = result === 'AGREE' ? 'alm-agree' : result === 'SPLIT' ? 'alm-split' : 'alm-neutral';
        return '<span class="' + className + '">' + escapeHtml(result) + '</span>';
    }

    function formatCurrency(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return 'N/A';
        }
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            maximumFractionDigits: 2,
        }).format(numeric);
    }

    function formatPercent(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return 'N/A';
        }
        const prefix = numeric > 0 ? '+' : '';
        return prefix + numeric.toFixed(1) + '%';
    }

    function formatPctDelta(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return 'N/A';
        }
        const percentValue = numeric * 100;
        const prefix = percentValue > 0 ? '+' : '';
        return prefix + percentValue.toFixed(2) + '%';
    }

    function formatAccuracyPct(value, digits = 1) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return '0.0';
        }
        return numeric.toFixed(digits);
    }

    function formatLongDate(dateStr) {
        const date = new Date(dateStr + 'T12:00:00');
        return date.toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'long',
            day: 'numeric',
            year: 'numeric',
        });
    }

    function formatMonthLabel(monthKey) {
        const date = new Date(monthKey + '-01T12:00:00');
        return date.toLocaleDateString('en-US', {
            month: 'long',
            year: 'numeric',
        });
    }

    function formatShortDay(dateStr) {
        const date = new Date(dateStr + 'T12:00:00');
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
        });
    }

    function formatMonthShort(monthKey) {
        return new Date(monthKey + '-01T12:00:00').toLocaleDateString('en-US', { month: 'short' });
    }

    function formatAlmanacDirection(direction) {
        const normalized = String(direction || '').toUpperCase();
        if (normalized === 'D') {
            return 'Bullish';
        }
        if (normalized === 'N') {
            return 'Bearish';
        }
        return 'Sideways';
    }

    function monthDayToneClass(day) {
        if (!day) {
            return 'is-empty';
        }
        if (!day.almanac_available) {
            return day.status === 'closed' ? 'is-closed' : 'is-no-data';
        }
        const scores = WEEKLY_INDEX_META
            .map((indexMeta) => Number(day[indexMeta.scoreKey]))
            .filter((value) => Number.isFinite(value));
        const average = scores.length
            ? scores.reduce((sum, value) => sum + value, 0) / scores.length
            : 50;
        if (average >= 60) {
            return 'is-bullish';
        }
        if (average <= 40) {
            return 'is-bearish';
        }
        return 'is-mixed';
    }

    function buildMonthlyCalendarScoreLine(day, indexMeta) {
        const score = day?.[indexMeta.scoreKey];
        if (score === null || score === undefined || score === '') {
            return '';
        }
        return ''
            + '<div class="alm-cal-line ' + scoreClass(score) + '">'
            + '  <span class="alm-cal-line-label">' + escapeHtml(indexMeta.shortLabel) + '</span>'
            + '  <strong>' + escapeHtml(score) + '</strong>'
            + '</div>';
    }

    function monthlyStatusBadge(day) {
        if (!day || day.almanac_available) {
            return '';
        }
        if (day.status === 'closed') {
            if (day.is_weekend) {
                return 'Weekend';
            }
            return 'Holiday';
        }
        return 'No Entry';
    }

    function buildMonthlyTooltipIndexRow(day, accDay, indexMeta) {
        const score = day?.[indexMeta.scoreKey];
        const direction = day?.[indexMeta.directionKey];
        const result = accDay?.results?.[indexMeta.accuracyKey] || {};
        const pctValue = Number(accDay?.pct_change?.[indexMeta.pctKey]);
        const pctClass = pctValue > 0 ? 'alm-acc-positive' : pctValue < 0 ? 'alm-acc-negative' : 'alm-note-caption';
        const actualText = accDay && Number.isFinite(pctValue)
            ? 'Actual ' + formatPctDelta(pctValue)
            : 'Historic result pending';
        const verdictText = result.verdict || 'No call';

        return ''
            + '<div class="alm-month-tip-row">'
            + '  <div class="alm-month-tip-row-head">'
            + '    <span class="alm-mini-title">' + escapeHtml(indexMeta.label) + '</span>'
            + '    <span class="alm-week-index-score ' + scoreClass(score) + '">' + escapeHtml(score) + '</span>'
            +      directionBadge(direction)
            + '  </div>'
            + '  <div class="alm-month-tip-row-meta">'
            + '    <span class="' + pctClass + '">' + escapeHtml(actualText) + '</span>'
            + '    <span class="alm-week-index-verdict ' + accuracyCardClass(result.verdict) + '">' + escapeHtml(verdictText) + '</span>'
            + '  </div>'
            + '</div>';
    }

    function showMonthlyTooltip(target) {
        if (!target) {
            return;
        }
        const dateKey = target.getAttribute('data-month-date') || '';
        const day = currentMonthCalendarLookup[dateKey];
        if (!day) {
            return;
        }

        const tooltip = ensureMonthlyTooltip();
        const accDay = accuracyCache?.daily?.[dateKey] || null;
        const statusLabel = day.almanac_available
            ? 'Almanac entry'
            : (day.status === 'closed' ? 'Market closed' : 'No Almanac entry');
        const noteText = day.almanac_available
            ? (day.notes || 'No additional Almanac note for this day.')
            : (day.status_reason || 'No additional detail is available for this date.');
        const iconText = day.icon ? capitalize(String(day.icon).replace(/_/g, ' ')) : '';
        const accuracyText = accDay && Number(accDay.total_calls) > 0
            ? 'Historic accuracy: ' + accDay.hits + '/' + accDay.total_calls + ' correct'
            : 'Historic accuracy: no scored result yet';

        resetTooltipPosition(tooltip);
        tooltip.innerHTML = ''
            + '<div class="alm-month-tooltip-title">' + escapeHtml(formatLongDate(dateKey)) + '</div>'
            + '<div class="alm-month-tooltip-meta">'
            + '  <span class="alm-month-tooltip-chip">' + escapeHtml(statusLabel) + '</span>'
            +      (iconText ? '<span class="alm-month-tooltip-chip is-icon">' + escapeHtml(iconText) + '</span>' : '')
            + '</div>'
            +      (day.almanac_available
                ? '<div class="alm-month-tip-grid">' + WEEKLY_INDEX_META.map((indexMeta) => buildMonthlyTooltipIndexRow(day, accDay, indexMeta)).join('') + '</div>'
                : '')
            + '<div class="alm-month-tip-note">' + escapeHtml(noteText) + '</div>'
            +      (day.almanac_available ? '<div class="alm-month-tip-accuracy">' + escapeHtml(accuracyText) + '</div>' : '');
        tooltip.classList.add('is-visible');
        tooltip.setAttribute('aria-hidden', 'false');
        activeMonthlyTooltipTarget = target;
        positionMonthlyTooltip(target);
    }

    function hideMonthlyTooltip() {
        if (!monthlyTooltip) {
            return;
        }
        monthlyTooltip.classList.remove('is-visible');
        monthlyTooltip.setAttribute('aria-hidden', 'true');
        resetTooltipPosition(monthlyTooltip);
        activeMonthlyTooltipTarget = null;
    }

    function accuracyCardClass(verdict) {
        if (verdict === 'HIT') {
            return 'alm-acc-hit';
        }
        if (verdict === 'MISS') {
            return 'alm-acc-miss';
        }
        return 'alm-acc-neutral';
    }

    function dailyAccuracyClass(accDay) {
        if (!accDay || Number(accDay.total_calls) <= 0) {
            return 'alm-acc-neutral';
        }
        if (Number(accDay.hits) === Number(accDay.total_calls)) {
            return 'alm-acc-hit';
        }
        if (Number(accDay.hits) === 0) {
            return 'alm-acc-miss';
        }
        return 'alm-acc-neutral';
    }

    function buildWeeklyAccuracyFallback(accPayload, weekDates) {
        const fallback = {
            hits: 0,
            total_calls: 0,
            accuracy: 0,
            dow: { hits: 0, total: 0, pct: 0 },
            sp500: { hits: 0, total: 0, pct: 0 },
            nasdaq: { hits: 0, total: 0, pct: 0 },
        };

        weekDates.forEach((dateKey) => {
            const accDay = accPayload?.daily?.[dateKey];
            if (!accDay) {
                return;
            }
            fallback.hits += Number(accDay.hits || 0);
            fallback.total_calls += Number(accDay.total_calls || 0);

            WEEKLY_INDEX_META.forEach((indexMeta) => {
                const result = accDay.results?.[indexMeta.accuracyKey];
                if (!result || !result.verdict) {
                    return;
                }
                fallback[indexMeta.weeklyKey].total += 1;
                if (result.verdict === 'HIT') {
                    fallback[indexMeta.weeklyKey].hits += 1;
                }
            });
        });

        fallback.accuracy = fallback.total_calls > 0
            ? Number(formatAccuracyPct((fallback.hits / fallback.total_calls) * 100, 1))
            : 0;

        WEEKLY_INDEX_META.forEach((indexMeta) => {
            const bucket = fallback[indexMeta.weeklyKey];
            bucket.pct = bucket.total > 0
                ? Number(formatAccuracyPct((bucket.hits / bucket.total) * 100, 1))
                : 0;
        });

        return fallback;
    }

    function buildWeeklyIndexCell(day, accDay, indexMeta) {
        const score = day?.[indexMeta.scoreKey];
        const direction = day?.[indexMeta.directionKey];
        const result = accDay?.results?.[indexMeta.accuracyKey] || {};
        const pctValue = Number(accDay?.pct_change?.[indexMeta.pctKey]);
        const pctClass = pctValue > 0 ? 'alm-acc-positive' : pctValue < 0 ? 'alm-acc-negative' : 'alm-note-caption';
        const verdictClass = accuracyCardClass(result.verdict);
        const verdictLabel = result.verdict || 'NO CALL';
        const predictionLabel = result.predicted || '--';
        const actualLabel = result.actual || '--';

        return ''
            + '<div class="alm-week-index-cell">'
            + '  <div class="alm-week-index-head">'
            + '    <span class="alm-week-index-score ' + scoreClass(score) + '">' + escapeHtml(score) + '</span>'
            +      directionBadge(direction)
            + '  </div>'
            + '  <div class="alm-week-index-meta">'
            + '    <span class="alm-week-index-trend">' + escapeHtml(predictionLabel) + ' call</span>'
            + '    <span class="alm-week-index-sep">&middot;</span>'
            + '    <span class="' + pctClass + '">Actual ' + escapeHtml(formatPctDelta(pctValue)) + '</span>'
            + '  </div>'
            + '  <div class="alm-week-index-foot">'
            + '    <span class="alm-week-index-verdict ' + verdictClass + '">' + escapeHtml(verdictLabel) + '</span>'
            + '    <span class="alm-note-caption">Actual ' + escapeHtml(actualLabel) + '</span>'
            + '  </div>'
            + '</div>';
    }

    function buildWeeklyStatusCell(day) {
        const closed = day?.status === 'closed';
        const label = closed ? 'Market Closed' : 'No Almanac Data';
        const chipClass = closed ? 'alm-week-status-chip is-closed' : 'alm-week-status-chip';
        const detail = closed ? 'No trading session' : 'Market open';

        return ''
            + '<div class="alm-week-index-cell alm-week-index-empty">'
            + '  <div class="alm-week-index-head">'
            + '    <span class="' + chipClass + '">' + escapeHtml(label) + '</span>'
            + '  </div>'
            + '  <div class="alm-week-index-meta">'
            + '    <span class="alm-note-caption">' + escapeHtml(detail) + '</span>'
            + '  </div>'
            + '</div>';
    }

    function buildWeeklySummaryCard(label, summary) {
        const hits = Number(summary?.hits || 0);
        const totalValue = summary?.total ?? summary?.total_calls ?? 0;
        const total = Number(totalValue);
        const pct = total > 0
            ? Number(summary?.pct ?? summary?.accuracy ?? ((hits / total) * 100))
            : 0;
        const toneClass = pct >= 60 ? 'acc-good' : pct < 40 ? 'acc-poor' : 'acc-mixed';

        return ''
            + '<div class="alm-agree-cell alm-week-summary-card">'
            + '  <div class="alm-weekday">' + escapeHtml(label) + '</div>'
            + '  <div class="alm-mini-value ' + toneClass + '">' + escapeHtml(hits + '/' + total) + '</div>'
            + '  <div class="alm-weekday">' + escapeHtml(formatAccuracyPct(pct, 0) + '% accuracy') + '</div>'
            + '</div>';
    }

    function buildAccuracyPanel(accDay, dateStr) {
        const cards = [
            { signalKey: 'd', dataKey: 'dji', label: 'Dow' },
            { signalKey: 's', dataKey: 'sp500', label: 'S&P 500' },
            { signalKey: 'n', dataKey: 'nasdaq', label: 'NASDAQ' },
        ].map((entry) => {
            const result = accDay?.results?.[entry.signalKey] || {};
            const pctValue = Number(accDay?.pct_change?.[entry.dataKey]);
            const pctClass = pctValue > 0 ? 'alm-acc-positive' : pctValue < 0 ? 'alm-acc-negative' : 'alm-note-caption';
            const verdictLabel = result.verdict || '--';
            const predictedLabel = result.predicted || '--';
            const actualLabel = result.actual || '--';
            return ''
                + '<div class="alm-acc-card ' + accuracyCardClass(result.verdict) + '">'
                + '  <div class="alm-mini-title">' + escapeHtml(entry.label) + '</div>'
                + '  <div><strong>' + escapeHtml(predictedLabel) + '</strong> predicted</div>'
                + '  <div class="' + pctClass + '"><strong>' + escapeHtml(formatPctDelta(pctValue)) + '</strong> actual</div>'
                + '  <div class="alm-note-caption">Actual direction: ' + escapeHtml(actualLabel) + '</div>'
                + '  <div class="alm-acc-badge">' + escapeHtml(verdictLabel) + '</div>'
                + '</div>';
        }).join('');

        const noteMarkup = accDay.context
            ? '<div class="alm-note-box">' + escapeHtml(accDay.context) + '</div>'
            : '';

        return ''
            + '<div class="alm-accuracy-panel">'
            + '  <div class="alm-mini-title">Backtest Result - ' + escapeHtml(formatLongDate(dateStr)) + '</div>'
            + '  <div class="alm-acc-grid">' + cards + '</div>'
            + '  <div class="alm-acc-score">' + escapeHtml(accDay.hits + '/' + accDay.total_calls + ' correct') + '</div>'
            + noteMarkup
            + '</div>';
    }

    function irisSignalClass(signal) {
        const normalized = String(signal || '').toUpperCase();
        if (normalized.includes('BUY')) {
            return 'alm-signal-buy';
        }
        if (normalized.includes('SELL')) {
            return 'alm-signal-sell';
        }
        return 'alm-signal-hold';
    }

    function irisDirectionArrow(direction) {
        if (direction === 'upward') {
            return '&uarr;';
        }
        if (direction === 'downward') {
            return '&darr;';
        }
        return '&rarr;';
    }

    function irisDirectionColor(direction) {
        if (direction === 'upward') {
            return 'var(--status-green)';
        }
        if (direction === 'downward') {
            return 'var(--status-red)';
        }
        return 'var(--text-muted)';
    }

    function formatIrisPctChange(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return 'N/A';
        }
        const prefix = numeric > 0 ? '+' : '';
        return prefix + numeric.toFixed(2) + '%';
    }

    function getPrimaryIrisSnapshot(indices) {
        return indices?.spy || indices?.gspc || indices?.dji || indices?.ixic || {};
    }

    function buildIrisTrendSummary(indices) {
        const order = [
            { key: 'dji', label: 'Dow' },
            { key: 'gspc', label: 'S&P 500' },
            { key: 'ixic', label: 'NASDAQ' },
        ];
        const parts = order
            .map((item) => {
                const entry = indices?.[item.key];
                if (!entry?.available || !entry.trend_label) {
                    return '';
                }
                return item.label + ': ' + entry.trend_label;
            })
            .filter(Boolean);
        return parts.join(' | ');
    }

    function ensureMonthAccuracySummary(calendar) {
        if (!calendar || !calendar.parentNode) {
            return null;
        }
        let summary = document.getElementById('monthly-calendar-accuracy-summary');
        if (!summary) {
            summary = document.createElement('div');
            summary.id = 'monthly-calendar-accuracy-summary';
            calendar.insertAdjacentElement('afterend', summary);
        }
        return summary;
    }

    function capitalize(value) {
        const text = String(value || '').trim();
        if (!text) {
            return '';
        }
        return text.charAt(0).toUpperCase() + text.slice(1);
    }

    function getSeasonalMonthKeys() {
        const preferred = SEASONAL_MONTH_ORDER.filter((monthKey) => monthKeys.includes(monthKey));
        const extras = monthKeys.filter((monthKey) => !SEASONAL_MONTH_ORDER.includes(monthKey));
        return preferred.concat(extras);
    }

    function getSeasonalSignalMeta(type) {
        return SEASONAL_SIGNAL_META[type] || SEASONAL_SIGNAL_META.default;
    }

    function getSeasonalPhase(monthKey) {
        if (SEASONAL_PHASES.bullish.has(monthKey)) {
            return {
                key: 'bullish',
                label: 'Best Six Months',
                tone: 'bull',
                note: 'Historically stronger seasonal phase',
            };
        }
        if (SEASONAL_PHASES.bearish.has(monthKey)) {
            return {
                key: 'bearish',
                label: 'Worst Six Months',
                tone: 'bear',
                note: 'Historically weaker seasonal phase',
            };
        }
        return {
            key: 'neutral',
            label: 'Seasonal Phase',
            tone: 'bull',
            note: 'Additional seasonal context',
        };
    }

    function getBiasClass(bias) {
        if (bias === 'bullish') {
            return 'alm-dir-d';
        }
        if (bias === 'bearish') {
            return 'alm-dir-n';
        }
        return 'alm-dir-s';
    }

    function renderBiasBadge(bias) {
        const label = capitalize(bias || 'mixed') || 'Mixed';
        return '<span class="alm-direction-badge alm-sig-bias ' + getBiasClass(bias) + '">' + escapeHtml(label) + '</span>';
    }

    function renderSeasonalLegend() {
        const legendOrder = ['timing', 'risk_indicator', 'weekly_timing', 'macro_context', 'default'];
        return legendOrder.map((typeKey) => {
            const meta = getSeasonalSignalMeta(typeKey);
            return ''
                + '<div class="alm-sig-legend-item">'
                + '  <span class="alm-sig-dot ' + meta.className + '" aria-hidden="true"></span>'
                + '  <span><strong>' + escapeHtml(meta.category) + '</strong> <span class="alm-note-caption"> ' + escapeHtml(meta.meaning) + '</span></span>'
                + '</div>';
        }).join('');
    }

    function ensureSeasonalTooltip() {
        if (seasonalTooltip) {
            return seasonalTooltip;
        }
        seasonalTooltip = document.createElement('div');
        seasonalTooltip.className = 'alm-sig-tooltip';
        seasonalTooltip.setAttribute('role', 'tooltip');
        seasonalTooltip.setAttribute('aria-hidden', 'true');
        document.body.appendChild(seasonalTooltip);
        return seasonalTooltip;
    }

    function ensureMonthlyTooltip() {
        if (monthlyTooltip) {
            return monthlyTooltip;
        }
        monthlyTooltip = document.createElement('div');
        monthlyTooltip.className = 'alm-month-tooltip';
        monthlyTooltip.setAttribute('role', 'tooltip');
        monthlyTooltip.setAttribute('aria-hidden', 'true');
        document.body.appendChild(monthlyTooltip);
        return monthlyTooltip;
    }

    function resetTooltipPosition(tooltip) {
        if (!tooltip) {
            return;
        }
        tooltip.style.left = '-9999px';
        tooltip.style.top = '-9999px';
    }

    function positionFloatingTooltip(tooltip, target) {
        if (!tooltip || !target) {
            return;
        }
        const targetRect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const spacing = 12;
        let left = targetRect.left + ((targetRect.width - tooltipRect.width) / 2);
        let top = targetRect.bottom + spacing;

        if (top + tooltipRect.height > viewportHeight - spacing) {
            top = targetRect.top - tooltipRect.height - spacing;
        }
        if (top < spacing) {
            top = Math.max(spacing, targetRect.bottom + spacing);
        }

        left = Math.max(spacing, Math.min(left, viewportWidth - tooltipRect.width - spacing));
        top = Math.max(spacing, Math.min(top, viewportHeight - tooltipRect.height - spacing));

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    function positionSeasonalTooltip(target) {
        positionFloatingTooltip(seasonalTooltip, target);
    }

    function positionMonthlyTooltip(target) {
        positionFloatingTooltip(monthlyTooltip, target);
    }

    function showSeasonalTooltip(target) {
        if (!target) {
            return;
        }
        const tooltip = ensureSeasonalTooltip();
        const title = target.getAttribute('data-sig-title') || target.textContent || 'Seasonal signal';
        const category = target.getAttribute('data-sig-category') || 'Reference';
        const description = target.getAttribute('data-sig-description') || 'No description available.';
        const month = target.getAttribute('data-sig-month') || '2026';
        const meaning = target.getAttribute('data-sig-meaning') || '';
        const toneClass = target.getAttribute('data-sig-tone') || 'sg-n';

        resetTooltipPosition(tooltip);
        tooltip.innerHTML = ''
            + '<div class="alm-sig-tooltip-title">' + escapeHtml(title) + '</div>'
            + '<div class="alm-sig-tooltip-meta">'
            + '  <span class="alm-sig-dot ' + escapeHtml(toneClass) + '" aria-hidden="true"></span>'
            + '  <span>' + escapeHtml(category) + ' | ' + escapeHtml(month) + '</span>'
            + '</div>'
            + '<div class="alm-sig-tooltip-body">' + escapeHtml(description) + '</div>'
            + (meaning ? '<div class="alm-sig-tooltip-meaning">' + escapeHtml(meaning) + '</div>' : '');
        tooltip.classList.add('is-visible');
        tooltip.setAttribute('aria-hidden', 'false');
        activeTooltipTarget = target;
        positionSeasonalTooltip(target);
    }

    function hideSeasonalTooltip() {
        if (!seasonalTooltip) {
            return;
        }
        seasonalTooltip.classList.remove('is-visible');
        seasonalTooltip.setAttribute('aria-hidden', 'true');
        resetTooltipPosition(seasonalTooltip);
        activeTooltipTarget = null;
    }

    function getSignalBadgeTarget(node) {
        return node instanceof Element ? node.closest('.alm-sig-badge') : null;
    }

    function getMonthDayTarget(node) {
        return node instanceof Element ? node.closest('.alm-cal-day[data-month-date]') : null;
    }

    function resolveNearestTradingDate(candidate) {
        if (!tradingDates.length) {
            return DEFAULT_DATE;
        }
        if (tradingDates.includes(candidate)) {
            return candidate;
        }
        const nextDate = tradingDates.find((dateKey) => dateKey >= candidate);
        if (nextDate) {
            return nextDate;
        }
        return tradingDates[tradingDates.length - 1];
    }

    function parseDateKey(dateStr) {
        return new Date(dateStr + 'T12:00:00');
    }

    function toDateKey(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return year + '-' + month + '-' + day;
    }

    function addDays(dateStr, amount) {
        const next = parseDateKey(dateStr);
        next.setDate(next.getDate() + amount);
        return toDateKey(next);
    }

    function getWeekStart(dateStr) {
        const date = parseDateKey(dateStr);
        const weekday = date.getDay();
        const offset = weekday === 0 ? -6 : 1 - weekday;
        date.setDate(date.getDate() + offset);
        return toDateKey(date);
    }

    function getAvailableWeekStarts() {
        return Array.from(new Set(tradingDates.map((dateKey) => getWeekStart(dateKey)))).sort();
    }

    function resolveNearestWeekStart(candidate) {
        const weekStarts = getAvailableWeekStarts();
        const desired = getWeekStart(candidate || DEFAULT_DATE);
        if (!weekStarts.length) {
            return desired;
        }
        if (weekStarts.includes(desired)) {
            return desired;
        }
        const nextWeek = weekStarts.find((weekStart) => weekStart >= desired);
        if (nextWeek) {
            return nextWeek;
        }
        return weekStarts[weekStarts.length - 1];
    }

    function getWeekDates(startDate) {
        if (!tradingDates.length) {
            return [];
        }
        const weekStart = resolveNearestWeekStart(startDate);
        const weekEnd = addDays(weekStart, 4);
        return tradingDates.filter((dateKey) => dateKey >= weekStart && dateKey <= weekEnd);
    }

    function formatWeekRangeLabel(weekStart) {
        const weekEnd = addDays(weekStart, 4);
        const startDate = parseDateKey(weekStart);
        const endDate = parseDateKey(weekEnd);
        const startOptions = { month: 'short', day: 'numeric' };
        const endOptions = { month: 'short', day: 'numeric', year: 'numeric' };

        if (startDate.getFullYear() !== endDate.getFullYear()) {
            startOptions.year = 'numeric';
        }

        return 'Week of '
            + startDate.toLocaleDateString('en-US', startOptions)
            + ' - '
            + endDate.toLocaleDateString('en-US', endOptions);
    }

    function getMonthDisplayRank(monthKey, monthInfo, heatmapEntry) {
        if (heatmapEntry && Number.isFinite(Number(heatmapEntry.sp500_midterm_rank))) {
            return Number(heatmapEntry.sp500_midterm_rank);
        }
        return Number(monthInfo?.vital_stats?.sp500?.rank || 0);
    }

    function getCurrentYearAwareDefaultDate() {
        if (!tradingDates.length) {
            return DEFAULT_DATE;
        }
        const today = new Date().toISOString().slice(0, 10);
        if (!today.startsWith('2026-')) {
            return resolveNearestTradingDate(DEFAULT_DATE);
        }
        return resolveNearestTradingDate(today);
    }

    function parseCheckEngineLight(rawValue) {
        const value = String(rawValue || '').trim();
        if (!value) {
            return { label: 'N/A — analyze a ticker first', tone: 'neutral', missing: true };
        }
        const tone = value.startsWith('RED') ? 'red' : value.startsWith('YELLOW') ? 'yellow' : 'green';
        return { label: value, tone, missing: false };
    }

    function getIrisSignal(payload) {
        const explicitSignal = String(payload?.signals?.investment_signal || '').trim().toUpperCase();
        if (explicitSignal) {
            return explicitSignal;
        }
        const trend = String(payload?.signals?.trend_label || '').toUpperCase();
        if (trend.includes('DOWNTREND')) {
            return 'SELL';
        }
        if (trend.includes('UPTREND')) {
            return 'BUY';
        }
        return 'HOLD';
    }

    function computeAgreement(irisSignal, almanacDirection) {
        const signal = String(irisSignal || '').toUpperCase();
        const direction = String(almanacDirection || '').toUpperCase();

        const irisBullish = signal.includes('BUY');
        const irisBearish = signal.includes('SELL');
        const almanacBullish = direction === 'D';
        const almanacBearish = direction === 'N';
        const almanacNeutral = direction === 'S';

        if ((irisBullish && almanacBullish) || (irisBearish && almanacBearish)) {
            return 'AGREE';
        }
        if ((irisBullish && almanacBearish) || (irisBearish && almanacBullish)) {
            return 'SPLIT';
        }
        if (signal === 'HOLD' || almanacNeutral) {
            return 'NEUTRAL';
        }
        return 'NEUTRAL';
    }

    function renderAlmanacError() {
        const message = '<div class="alm-empty">Almanac data could not be loaded.</div>';
        const dailyPanel = document.getElementById('almanac-daily-content');
        const weeklyTable = document.getElementById('weekly-table-container');
        const monthlyPanel = document.getElementById('almanac-monthly-content');
        const seasonalStrip = document.getElementById('seasonal-heatmap-strip');
        const seasonalLegend = document.getElementById('seasonal-legend');
        const seasonalSignals = document.getElementById('seasonal-signals-table');
        const seasonalRanking = document.getElementById('seasonal-monthly-ranking');
        if (dailyPanel) dailyPanel.innerHTML = message;
        if (weeklyTable) weeklyTable.innerHTML = message;
        if (monthlyPanel) monthlyPanel.innerHTML = message;
        if (seasonalStrip) seasonalStrip.innerHTML = message;
        if (seasonalLegend) seasonalLegend.innerHTML = '';
        if (seasonalSignals) seasonalSignals.innerHTML = '';
        if (seasonalRanking) seasonalRanking.innerHTML = '';
        hideSeasonalTooltip();
    }

    function monthMetricValueClass(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return '';
        }
        if (numeric > 0) {
            return 'is-positive';
        }
        if (numeric < 0) {
            return 'is-negative';
        }
        return '';
    }

    async function renderDailyView() {
        const payload = await loadAlmanacFull();
        const almanacPanel = document.getElementById('almanac-daily-content');
        const irisPanel = document.getElementById('iris-daily-content');
        const verdict = document.getElementById('daily-verdict');
        const dateLabel = document.getElementById('date-label');
        const datePicker = document.getElementById('date-picker');

        if (!payload) {
            renderAlmanacError();
            if (irisPanel) irisPanel.innerHTML = '<div class="alm-empty">IRIS comparison unavailable because almanac data failed to load.</div>';
            if (verdict) verdict.innerHTML = '<span class="alm-neutral">Comparison unavailable</span>';
            return;
        }

        currentDate = resolveNearestTradingDate(currentDate);
        currentWeekStart = getWeekStart(currentDate);
        currentMonth = currentDate.slice(0, 7);

        if (dateLabel) {
            dateLabel.textContent = formatLongDate(currentDate);
        }
        if (datePicker) {
            datePicker.value = currentDate;
        }

        const dayData = payload.daily[currentDate];
        if (!dayData) {
            if (almanacPanel) almanacPanel.innerHTML = '<div class="alm-empty">No Almanac data for this trading day.</div>';
            if (irisPanel) irisPanel.innerHTML = '<div class="alm-empty">No comparison available for this trading day.</div>';
            if (verdict) verdict.innerHTML = '<span class="alm-neutral">No comparison available</span>';
            return;
        }

        const notesMarkup = dayData.notes
            ? '<div class="alm-note-box">' + escapeHtml(dayData.notes) + '</div>'
            : '<div class="alm-note-box">No special session note recorded for this date.</div>';
        const iconMarkup = dayData.icon
            ? '<span class="alm-chip">Icon: ' + escapeHtml(dayData.icon.replace('_', ' ')) + '</span>'
            : '';

        if (almanacPanel) {
            almanacPanel.innerHTML = ''
                + '<div class="alm-score-shell">'
                + '  <div class="alm-score-value ' + scoreClass(dayData.s) + '">' + escapeHtml(dayData.s) + '</div>'
                + '  <div class="alm-score-caption">S&amp;P 500 historical up-day probability</div>'
                + '  <div class="alm-note-caption" style="margin-top:0.25rem;">'
                + '    <span style="color:var(--status-green);">60–100 Bullish</span>'
                + '    &nbsp;&middot;&nbsp;<span>41–59 Sideways</span>'
                + '    &nbsp;&middot;&nbsp;<span style="color:var(--status-red);">0–40 Bearish</span>'
                + '  </div>'
                + '</div>'
                + '<div class="alm-pill-row">'
                +      directionBadge(dayData.s_dir)
                +      iconMarkup
                + '</div>'
                + '<div class="alm-triple-grid">'
                + '  <div class="alm-mini-card">'
                + '    <div class="alm-mini-title">Dow</div>'
                + '    <div class="alm-mini-value ' + scoreClass(dayData.d) + '">' + escapeHtml(dayData.d) + '</div>'
                +       directionBadge(dayData.d_dir)
                + '  </div>'
                + '  <div class="alm-mini-card">'
                + '    <div class="alm-mini-title">S&amp;P 500</div>'
                + '    <div class="alm-mini-value ' + scoreClass(dayData.s) + '">' + escapeHtml(dayData.s) + '</div>'
                +       directionBadge(dayData.s_dir)
                + '  </div>'
                + '  <div class="alm-mini-card">'
                + '    <div class="alm-mini-title">NASDAQ</div>'
                + '    <div class="alm-mini-value ' + scoreClass(dayData.n) + '">' + escapeHtml(dayData.n) + '</div>'
                +       directionBadge(dayData.n_dir)
                + '  </div>'
                + '</div>'
                + notesMarkup;
        }

        const accPayload = await loadAccuracyData();
        const accDay = accPayload?.daily?.[currentDate];
        if (accDay && almanacPanel) {
            const accHtml = buildAccuracyPanel(accDay, currentDate);
            almanacPanel.insertAdjacentHTML('beforeend', accHtml);
        }

        if (irisPanel) {
            irisPanel.innerHTML = '<div class="alm-empty">Loading IRIS index snapshot...</div>';
        }

        const irisData = await fetchIrisLive();
        if (!irisData || !irisData.indices) {
            if (irisPanel) {
                irisPanel.innerHTML = ''
                    + '<div class="alm-empty">IRIS index snapshot is unavailable right now.'
                    + '<br><small>Use the refresh action once report files are available.</small></div>';
            }
            if (verdict) {
                verdict.innerHTML = '<span class="alm-neutral">Almanac loaded; IRIS index snapshot unavailable</span>';
            }
            return;
        }
        if (irisPanel) {
            const indices = irisData.indices;
            const spyData = getPrimaryIrisSnapshot(indices);
            const light = parseCheckEngineLight(spyData.check_engine_light || '');
            const lightColor = light.tone === 'red'
                ? 'var(--status-red)'
                : light.tone === 'yellow'
                    ? 'var(--status-yellow)'
                    : light.tone === 'neutral'
                        ? 'var(--text-muted)'
                        : 'var(--status-green)';
            const indexCards = [
                { key: 'dji', label: 'Dow', data: indices.dji },
                { key: 'gspc', label: 'S&P 500', data: indices.gspc },
                { key: 'ixic', label: 'NASDAQ', data: indices.ixic },
            ].map((idx) => {
                const entry = idx.data;
                if (!entry || !entry.available) {
                    return '<div class="alm-mini-card"><div class="alm-mini-title">' + idx.label
                        + '</div><div class="alm-note-caption">No report — click Refresh</div></div>';
                }
                const direction = String(entry.direction || '').toLowerCase();
                const signalText = entry.investment_signal || (entry.trend_label ? '' : 'Loading...');
                return ''
                    + '<div class="alm-mini-card">'
                    + '  <div class="alm-mini-title">' + escapeHtml(idx.label) + '</div>'
                    + '  <div class="alm-mini-value" style="color:' + irisDirectionColor(direction) + ';">'
                    +       irisDirectionArrow(direction) + ' ' + escapeHtml(formatIrisPctChange(entry.pct_change))
                    + '  </div>'
                    + '  <div class="' + irisSignalClass(entry.investment_signal) + '">' + escapeHtml(signalText || 'HOLD') + '</div>'
                    + '  <div class="alm-note-caption">' + escapeHtml(entry.trend_label || 'Trend unavailable') + '</div>'
                    + '</div>';
            }).join('');

            const sessionDate = spyData.session_date || '';
            const today = new Date().toISOString().slice(0, 10);
            const isStale = Boolean(sessionDate) && sessionDate < today;
            const trendSummary = buildIrisTrendSummary(indices);
            const refreshMeta = sessionDate
                ? (isStale ? 'Data from: ' + sessionDate : 'Snapshot date: ' + sessionDate)
                : 'Snapshot date unavailable';
            const trendBox = trendSummary
                ? '<div class="alm-note-box"><strong>Trend Summary:</strong> ' + escapeHtml(trendSummary) + '</div>'
                : '';

            irisPanel.innerHTML = ''
                + '<div class="alm-score-shell">'
                + '  <div class="alm-score-caption">Check Engine Light</div>'
                + '  <div class="alm-score-value" style="color:' + lightColor + '; font-size:' + (light.missing ? '0.9rem' : '') + ';">'
                +       escapeHtml(light.missing ? 'No Report' : (light.label.split(' ')[0] || 'N/A'))
                + '  </div>'
                +      (light.missing ? '<div class="alm-note-caption" style="margin-top:0.25rem;">Click Refresh below to fetch live data</div>' : '')
                + '</div>'
                + '<div class="alm-triple-grid">' + indexCards + '</div>'
                + trendBox
                + '<div class="alm-note-caption" style="margin-top:0.35rem;">'
                + escapeHtml(refreshMeta)
                + ' | <a href="#" id="iris-refresh-btn" class="alm-refresh-link">Refresh</a></div>';

            const refreshBtn = document.getElementById('iris-refresh-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', async (event) => {
                    event.preventDefault();
                    refreshBtn.textContent = 'Refreshing...';
                    try {
                        const fresh = await fetchJson('/api/almanac/iris-refresh');
                        irisLiveData = fresh;
                        irisPromise = null;
                        renderDailyView();
                    } catch (error) {
                        console.error('IRIS refresh failed:', error);
                        refreshBtn.textContent = 'Refresh failed';
                    }
                });
            }
        }

        if (verdict && irisData?.indices) {
            const pairs = [
                { label: 'Dow', irisKey: 'dji', almDir: dayData.d_dir },
                { label: 'S&P', irisKey: 'gspc', almDir: dayData.s_dir },
                { label: 'NASDAQ', irisKey: 'ixic', almDir: dayData.n_dir },
            ];

            const badges = pairs.map((pair) => {
                const irisIdx = irisData.indices[pair.irisKey];
                if (!irisIdx?.available) {
                    return '<span class="alm-verdict-neutral">' + escapeHtml(pair.label) + ': --</span>';
                }

                const irisDir = irisIdx.direction === 'upward'
                    ? 'D'
                    : irisIdx.direction === 'downward'
                        ? 'N'
                        : 'S';
                const almDir = pair.almDir;
                const agree = (irisDir === 'D' && almDir === 'D') || (irisDir === 'N' && almDir === 'N');
                const agreeLabel = almDir === 'S' || irisDir === 'S' ? 'NEUTRAL' : (agree ? 'AGREE' : 'DISAGREE');

                return '<span class="alm-verdict-' + agreeLabel.toLowerCase() + '">'
                    + escapeHtml(pair.label) + ': ' + escapeHtml(agreeLabel) + '</span>';
            }).join(' <span class="alm-note-caption">|</span> ');

            verdict.innerHTML = ''
                + '<div class="alm-note-caption">IRIS direction vs Almanac historical bias'
                + ' &mdash; <strong>AGREE</strong> = both point the same way'
                + ' &nbsp;&middot;&nbsp; <strong>DISAGREE</strong> = opposing signals'
                + ' &nbsp;&middot;&nbsp; <strong>NEUTRAL</strong> = one or both are sideways / no clear call'
                + '</div>'
                + '<div style="margin-top:0.35rem;">' + badges + '</div>';
        }
    }

    async function renderWeeklyView() {
        const payload = await loadAlmanacFull();
        const weekLabel = document.getElementById('week-label');
        const container = document.getElementById('weekly-table-container');
        const strip = document.getElementById('weekly-agreement-strip');

        if (!payload) {
            if (container) container.innerHTML = '<div class="alm-empty">Weekly Almanac data could not be loaded.</div>';
            if (strip) strip.innerHTML = '';
            return;
        }

        const accPayload = await loadAccuracyData();
        currentWeekStart = resolveNearestWeekStart(currentWeekStart);
        let weekPayload = null;
        try {
            weekPayload = await fetchJson('/api/almanac/week?start=' + encodeURIComponent(currentWeekStart));
        } catch (error) {
            console.error('Failed to load weekly almanac view:', error);
        }
        if (!weekPayload || !Array.isArray(weekPayload.weekdays) || !weekPayload.weekdays.length) {
            if (container) container.innerHTML = '<div class="alm-empty">No weekly calendar data found for that start date.</div>';
            if (strip) strip.innerHTML = '';
            return;
        }

        const weekDays = weekPayload.weekdays;
        const weekDates = weekDays.filter((day) => day.almanac_available).map((day) => day.date);
        currentWeekStart = weekPayload.week_start || currentWeekStart;
        if (weekLabel) {
            weekLabel.textContent = formatWeekRangeLabel(currentWeekStart);
        }

        const weekAccuracyFromApi = await loadWeekAccuracy(currentWeekStart);
        const weekAccuracy = weekAccuracyFromApi || buildWeeklyAccuracyFallback(accPayload, weekDates);
        const usingFallbackAccuracy = !weekAccuracyFromApi;

        if (container) {
            const rows = weekDays.map((day) => {
                const acc = day.almanac_available ? accPayload?.daily?.[day.date] : null;
                const accClass = dailyAccuracyClass(acc);
                const indexCells = WEEKLY_INDEX_META.map((indexMeta) => {
                    const cellMarkup = day.almanac_available
                        ? buildWeeklyIndexCell(day, acc, indexMeta)
                        : buildWeeklyStatusCell(day);
                    return '<td>' + cellMarkup + '</td>';
                }).join('');
                const iconMarkup = day.almanac_available
                    ? (day.icon ? '<span class="alm-chip">' + escapeHtml(day.icon.replace('_', ' ')) + '</span>' : '<span class="alm-note-caption">None</span>')
                    : '<span class="alm-chip">' + escapeHtml(day.status === 'closed' ? 'Holiday / Closure' : 'Open / No Entry') + '</span>';
                const noteText = day.almanac_available ? (day.notes || '-') : (day.status_reason || 'No additional detail available.');
                const accuracyPct = acc && Number(acc.total_calls) > 0
                    ? formatAccuracyPct((Number(acc.hits || 0) / Number(acc.total_calls || 1)) * 100, 0) + '% accuracy'
                    : (day.status === 'closed' ? 'Market closed' : 'No scoring data');
                const accuracyCell = acc
                    ? ''
                        + '<td class="alm-week-accuracy-cell">'
                        + '  <div class="alm-week-index-verdict ' + accClass + '">' + escapeHtml(acc.hits + '/' + acc.total_calls) + '</div>'
                        + '  <div class="alm-note-caption">' + escapeHtml(accuracyPct) + '</div>'
                        + '</td>'
                    : '<td class="alm-week-accuracy-cell"><span class="alm-note-caption">' + escapeHtml(accuracyPct) + '</span></td>';
                return ''
                    + '<tr>'
                    + '  <td><strong>' + escapeHtml(day.day) + '</strong><br><span class="alm-note-caption">' + escapeHtml(day.date) + '</span></td>'
                    + indexCells
                    + '  <td>' + iconMarkup + '</td>'
                    + '  <td>' + escapeHtml(noteText) + '</td>'
                    + accuracyCell
                    + '</tr>';
            }).join('');

            container.innerHTML = ''
                + '<table class="alm-week-table">'
                + '  <thead><tr><th>Day</th><th>Dow</th><th>S&amp;P 500</th><th>NASDAQ</th><th>Icon</th><th>Notes</th><th>Backtest (Hits/Total)</th></tr></thead>'
                + '  <tbody>' + rows + '</tbody>'
                + '</table>';
        }

        if (strip) {
            const fallbackNote = usingFallbackAccuracy
                ? '<div class="alm-note-caption" style="width:100%;text-align:center;margin-top:0.5rem;">* Accuracy totals computed from daily records (no weekly summary available for this period)</div>'
                : '';
            strip.innerHTML = ''
                + buildWeeklySummaryCard('Overall', weekAccuracy)
                + buildWeeklySummaryCard('Dow', weekAccuracy?.dow)
                + buildWeeklySummaryCard('S&P 500', weekAccuracy?.sp500)
                + buildWeeklySummaryCard('NASDAQ', weekAccuracy?.nasdaq)
                + fallbackNote;
        }
    }

    async function renderMonthlyView() {
        const [payload, accPayload] = await Promise.all([loadAlmanacFull(), loadAccuracyData()]);
        const monthLabel = document.getElementById('month-label');
        const almanacPanel = document.getElementById('almanac-monthly-content');
        const irisPanel = document.getElementById('iris-monthly-content');
        const calendar = document.getElementById('monthly-calendar-heatmap');
        const monthSummary = ensureMonthAccuracySummary(calendar);
        currentMonthCalendarLookup = {};
        hideMonthlyTooltip();

        if (monthSummary) {
            monthSummary.style.display = 'none';
            monthSummary.innerHTML = '';
        }

        if (!payload) {
            if (almanacPanel) almanacPanel.innerHTML = '<div class="alm-empty">Monthly Almanac data could not be loaded.</div>';
            if (irisPanel) irisPanel.innerHTML = '<div class="alm-empty">Monthly IRIS summary unavailable.</div>';
            if (calendar) calendar.innerHTML = '<div class="alm-empty">Calendar unavailable.</div>';
            return;
        }

        if (!monthKeys.includes(currentMonth)) {
            currentMonth = monthKeys.includes('2026-04') ? '2026-04' : monthKeys[0];
        }

        const monthPayload = await loadMonthDetail(currentMonth);
        const monthInfo = monthPayload?.month || payload.months[currentMonth];
        if (!monthInfo) {
            if (almanacPanel) almanacPanel.innerHTML = '<div class="alm-empty">No monthly data found.</div>';
            if (calendar) calendar.innerHTML = '<div class="alm-empty">Calendar unavailable for this month.</div>';
            return;
        }

        if (monthLabel) {
            monthLabel.textContent = formatMonthLabel(currentMonth);
        }

        const sp500 = monthInfo.vital_stats?.sp500 || {};
        if (almanacPanel) {
            almanacPanel.innerHTML = ''
                + '<div class="alm-month-metrics">'
                + '  <div class="alm-metric-card">'
                + '    <div class="alm-stat-label">S&amp;P Rank</div>'
                + '    <div class="alm-metric-value">#' + escapeHtml(sp500.rank ?? '?') + '</div>'
                + '  </div>'
                + '  <div class="alm-metric-card">'
                + '    <div class="alm-stat-label">All-Year Avg</div>'
                + '    <div class="alm-metric-value ' + monthMetricValueClass(sp500.avg_change) + '">' + escapeHtml(formatPercent(sp500.avg_change)) + '</div>'
                + '  </div>'
                + '  <div class="alm-metric-card">'
                + '    <div class="alm-stat-label">Midterm Avg</div>'
                + '    <div class="alm-metric-value ' + monthMetricValueClass(sp500.midterm_avg) + '">' + escapeHtml(formatPercent(sp500.midterm_avg)) + '</div>'
                + '  </div>'
                + '</div>'
                + '<div class="alm-stat-grid">'
                + '  <div class="alm-stat-card">'
                + '    <div class="alm-stat-label">Up Years</div>'
                + '    <div class="alm-stat-value">' + escapeHtml(sp500.up ?? 0) + '</div>'
                + '  </div>'
                + '  <div class="alm-stat-card">'
                + '    <div class="alm-stat-label">Down Years</div>'
                + '    <div class="alm-stat-value">' + escapeHtml(sp500.down ?? 0) + '</div>'
                + '  </div>'
                + '</div>'
                + '<div class="alm-note-box">' + escapeHtml(monthInfo.overview || 'No overview available for this month.') + '</div>';
        }

        if (irisPanel) {
            irisPanel.innerHTML = ''
                + '<div class="alm-note-box">'
                + '<strong>IRIS day-by-day view</strong><br><br>'
                + 'IRIS predictions are compared against the Almanac one trading day at a time. '
                + 'Switch to the <strong>Daily</strong> tab and use the date arrows to step through any trading day in this month.'
                + '<br><br>'
                + 'The calendar below shows Almanac historical probability scores for each day in ' + escapeHtml(formatMonthLabel(currentMonth)) + '. '
                + 'Hover or tap any date cell to see the full breakdown.'
                + '</div>';
        }

        if (calendar) {
            const calendarDays = Array.isArray(monthPayload?.calendar_days) ? monthPayload.calendar_days : [];
            currentMonthCalendarLookup = calendarDays.reduce((lookup, day) => {
                lookup[day.date] = day;
                return lookup;
            }, {});
            const monthStart = new Date(currentMonth + '-01T12:00:00');
            const mondayOffset = (monthStart.getDay() + 6) % 7;
            const pieces = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((label) => {
                return '<div class="alm-weekday" style="text-align:center; padding:0.2rem;">' + label + '</div>';
            });

            for (let index = 0; index < mondayOffset; index += 1) {
                pieces.push('<div class="alm-cal-day empty"></div>');
            }

            if (!calendarDays.length) {
                calendar.innerHTML = '<div class="alm-empty">No calendar day data found for this month.</div>';
            } else {
                calendarDays.forEach((dayData) => {
                    const dayNumber = Number(String(dayData.date || '').slice(-2));
                    const accDay = dayData.almanac_available ? accPayload?.daily?.[dayData.date] : null;
                    const accuracyClass = dailyAccuracyClass(accDay);
                    const statusBadge = dayData.almanac_available
                        ? (dayData.icon ? '<span class="alm-cal-badge">' + escapeHtml(capitalize(String(dayData.icon).replace(/_/g, ' '))) + '</span>' : '')
                        : '<span class="alm-cal-badge is-muted">' + escapeHtml(monthlyStatusBadge(dayData)) + '</span>';
                    const linesMarkup = dayData.almanac_available
                        ? WEEKLY_INDEX_META.map((indexMeta) => buildMonthlyCalendarScoreLine(dayData, indexMeta)).join('')
                        : ''
                            + '<div class="alm-cal-status">' + escapeHtml(dayData.status === 'closed' ? 'Market Closed' : 'No Almanac Entry') + '</div>'
                            + '<div class="alm-cal-status-sub">' + escapeHtml(dayData.is_weekend ? 'Weekend' : (dayData.status === 'closed' ? 'Holiday' : 'Market open')) + '</div>';
                    const accuracyDot = accDay && Number(accDay.total_calls) > 0
                        ? '<span class="alm-cal-acc-dot ' + accuracyClass + '"></span>'
                        : '';
                    pieces.push(
                        ''
                        + '<div class="alm-cal-day ' + monthDayToneClass(dayData) + '" tabindex="0" data-month-date="' + escapeHtml(dayData.date) + '" aria-label="' + escapeHtml(formatLongDate(dayData.date)) + '">'
                        + '  <div class="alm-cal-daytop">'
                        + '    <div class="alm-cal-daynum">' + escapeHtml(dayNumber) + '</div>'
                        +      statusBadge
                        + '  </div>'
                        + '  <div class="alm-cal-body">' + linesMarkup + '</div>'
                        +      accuracyDot
                        + '</div>'
                    );
                });

                calendar.innerHTML = pieces.join('');
            }
        }

        if (monthSummary) {
            const monthAcc = accPayload?.monthly?.[currentMonth];
            if (monthAcc) {
                monthSummary.style.display = '';
                monthSummary.className = 'alm-month-acc-summary';
                monthSummary.innerHTML = 'Month Accuracy: '
                    + monthAcc.hits + '/' + monthAcc.total_calls + ' direction calls correct (' + formatAccuracyPct(monthAcc.accuracy, 1) + '%)'
                    + '<span class="alm-note-caption"> Dow: ' + formatAccuracyPct(monthAcc.dow?.pct ?? 0, 0) + '%'
                    + ' &middot; S&amp;P: ' + formatAccuracyPct(monthAcc.sp500?.pct ?? 0, 0) + '%'
                    + ' &middot; NASDAQ: ' + formatAccuracyPct(monthAcc.nasdaq?.pct ?? 0, 0) + '%</span>';
            }
        }
    }

    async function renderSeasonalView() {
        const payload = await loadAlmanacFull();
        const strip = document.getElementById('seasonal-heatmap-strip');
        const legend = document.getElementById('seasonal-legend');
        const signalsTable = document.getElementById('seasonal-signals-table');
        const ranking = document.getElementById('seasonal-monthly-ranking');

        if (!payload) {
            if (strip) strip.innerHTML = '<div class="alm-empty">Seasonal data could not be loaded.</div>';
            if (legend) legend.innerHTML = '';
            if (signalsTable) signalsTable.innerHTML = '';
            if (ranking) ranking.innerHTML = '';
            hideSeasonalTooltip();
            return;
        }

        const seasonalMonthKeys = getSeasonalMonthKeys();
        const signalsByMonth = {};
        seasonalMonthKeys.forEach((monthKey) => {
            signalsByMonth[monthKey] = [];
        });
        (Array.isArray(payload.signals) ? payload.signals : []).forEach((signal) => {
            const monthKey = signal?.source_month;
            if (!monthKey) {
                return;
            }
            if (!signalsByMonth[monthKey]) {
                signalsByMonth[monthKey] = [];
            }
            signalsByMonth[monthKey].push(signal);
        });

        if (strip) {
            strip.innerHTML = seasonalMonthKeys.map((monthKey) => {
                const monthInfo = payload.months[monthKey];
                const heatmapEntry = payload.heatmap[monthKey] || {};
                const rankValue = getMonthDisplayRank(monthKey, monthInfo, heatmapEntry);
                const isCurrent = monthKey === currentMonth;
                const monthLabel = formatMonthShort(monthKey);
                const tooltip = monthLabel + ' 2026 | overall rank #' + (monthInfo?.vital_stats?.sp500?.rank ?? '?')
                    + ' | midterm rank #' + (heatmapEntry.sp500_midterm_rank ?? '?')
                    + ' | midterm avg ' + formatPercent(heatmapEntry.sp500_midterm);
                return ''
                    + '<div class="alm-month-cell ' + escapeHtml(heatmapEntry.bias || 'mixed') + (isCurrent ? ' current' : '') + '" title="' + escapeHtml(tooltip) + '">'
                    + '  <div>' + escapeHtml(monthLabel) + '</div>'
                    + '  <div class="alm-rank-meta">#' + escapeHtml(rankValue || '?') + '</div>'
                    + '</div>';
            }).join('');
        }

        if (legend) {
            legend.innerHTML = renderSeasonalLegend();
        }

        if (signalsTable) {
            const rows = [];
            let previousPhaseKey = '';

            seasonalMonthKeys.forEach((monthKey) => {
                const monthInfo = payload.months[monthKey];
                if (!monthInfo) {
                    return;
                }
                const phase = getSeasonalPhase(monthKey);
                const heatmapEntry = payload.heatmap[monthKey] || {};
                const sp500 = monthInfo.vital_stats?.sp500 || {};
                const monthSignals = (signalsByMonth[monthKey] || [])
                    .slice()
                    .sort((left, right) => {
                        const typeSort = getSeasonalSignalMeta(left.type).order - getSeasonalSignalMeta(right.type).order;
                        if (typeSort !== 0) {
                            return typeSort;
                        }
                        return String(left.label || '').localeCompare(String(right.label || ''));
                    });

                if (phase.key !== previousPhaseKey) {
                    rows.push(
                        '<tr class="alm-sig-phase-row">'
                        + '  <td colspan="5">'
                        + '    <span class="alm-sig-phase ' + phase.tone + '">' + escapeHtml(phase.label) + '</span>'
                        + '    <span class="alm-sig-phase-note">' + escapeHtml(phase.note) + '</span>'
                        + '  </td>'
                        + '</tr>'
                    );
                    previousPhaseKey = phase.key;
                }

                const signalMarkup = monthSignals.length
                    ? '<div class="alm-sig-badges">' + monthSignals.map((signal) => {
                        const meta = getSeasonalSignalMeta(signal.type);
                        return ''
                            + '<button type="button" class="alm-sig-badge ' + meta.className + '"'
                            + ' data-sig-title="' + escapeHtml(signal.label || 'Seasonal signal') + '"'
                            + ' data-sig-category="' + escapeHtml(meta.category) + '"'
                            + ' data-sig-description="' + escapeHtml(signal.description || 'No description available in the current source.') + '"'
                            + ' data-sig-month="' + escapeHtml((monthInfo.name || formatMonthShort(monthKey)) + ' 2026') + '"'
                            + ' data-sig-meaning="' + escapeHtml(meta.meaning) + '"'
                            + ' data-sig-tone="' + escapeHtml(meta.className) + '">'
                            + escapeHtml(signal.label || 'Signal')
                            + '</button>';
                    }).join('') + '</div>'
                    : '<div class="alm-sig-empty">No tagged seasonal signals were extracted for this month in the current dataset.</div>';

                rows.push(
                    '<tr' + (monthKey === currentMonth ? ' class="alm-sig-current"' : '') + '>'
                    + '  <td class="alm-sig-col-month">'
                    + '    <span class="alm-sig-month-name">' + escapeHtml(monthInfo.name || monthKey) + '</span>'
                    + '    <span class="alm-sig-month-meta">Overall #' + escapeHtml(sp500.rank ?? '?') + '</span>'
                    + '  </td>'
                    + '  <td class="alm-sig-col-bias">' + renderBiasBadge(heatmapEntry.bias || 'mixed') + '</td>'
                    + '  <td class="alm-sig-col-rank alm-sig-hide-mobile"><span class="alm-sig-stat">#' + escapeHtml(getMonthDisplayRank(monthKey, monthInfo, heatmapEntry) || '?') + '</span></td>'
                    + '  <td class="alm-sig-col-midterm alm-sig-hide-mobile"><span class="alm-sig-stat ' + monthMetricValueClass(heatmapEntry.sp500_midterm) + '">' + escapeHtml(formatPercent(heatmapEntry.sp500_midterm)) + '</span></td>'
                    + '  <td class="alm-sig-col-signals">' + signalMarkup + '</td>'
                    + '</tr>'
                );
            });

            signalsTable.innerHTML = rows.length
                ? ''
                    + '<table class="alm-sig-table">'
                    + '  <thead><tr>'
                    + '    <th class="alm-sig-col-month">Month</th>'
                    + '    <th class="alm-sig-col-bias">Bias</th>'
                    + '    <th class="alm-sig-col-rank alm-sig-hide-mobile">Midterm Rank</th>'
                    + '    <th class="alm-sig-col-midterm alm-sig-hide-mobile">Midterm Avg</th>'
                    + '    <th class="alm-sig-col-signals">Signals</th>'
                    + '  </tr></thead>'
                    + '  <tbody>' + rows.join('') + '</tbody>'
                    + '</table>'
                : '<div class="alm-note-box">No tagged seasonal signals were extracted for this dataset.</div>';
        }

        if (ranking) {
            const rows = monthKeys
                .map((monthKey) => {
                    const monthInfo = payload.months[monthKey];
                    const sp500 = monthInfo?.vital_stats?.sp500 || {};
                    const heatmapEntry = payload.heatmap[monthKey] || {};
                    return {
                        name: monthInfo?.name || monthKey,
                        rank: Number(sp500.rank || 999),
                        avgChange: Number(sp500.avg_change || 0),
                        midtermAvg: Number(sp500.midterm_avg || 0),
                        midtermRank: Number(heatmapEntry.sp500_midterm_rank || 999),
                        up: Number(sp500.up || 0),
                        down: Number(sp500.down || 0),
                    };
                })
                .sort((left, right) => left.rank - right.rank)
                .map((row) => {
                    return ''
                        + '<tr>'
                        + '  <td><strong>#' + escapeHtml(row.rank) + '</strong></td>'
                        + '  <td>' + escapeHtml(row.name) + '</td>'
                        + '  <td class="' + monthMetricValueClass(row.avgChange) + '"><strong>' + escapeHtml(formatPercent(row.avgChange)) + '</strong></td>'
                        + '  <td class="' + monthMetricValueClass(row.midtermAvg) + '">' + escapeHtml(formatPercent(row.midtermAvg)) + '</td>'
                        + '  <td>#' + escapeHtml(row.midtermRank) + '</td>'
                        + '  <td>' + escapeHtml(row.up) + '/' + escapeHtml(row.up + row.down) + '</td>'
                        + '</tr>';
                })
                .join('');

            ranking.innerHTML = ''
                + '<div class="alm-rank-heading">S&amp;P 500 Monthly Ranking</div>'
                + '<table class="alm-week-table">'
                + '  <thead><tr><th>Overall Rank</th><th>Month</th><th>Avg Change</th><th>Midterm Avg</th><th>Midterm Rank</th><th>Win Rate</th></tr></thead>'
                + '  <tbody>' + rows + '</tbody>'
                + '</table>';
        }

        hideSeasonalTooltip();
    }

    function refreshCurrentView() {
        hideSeasonalTooltip();
        hideMonthlyTooltip();
        if (currentView === 'daily') {
            renderDailyView();
        } else if (currentView === 'weekly') {
            renderWeeklyView();
        } else if (currentView === 'monthly') {
            renderMonthlyView();
        } else {
            renderSeasonalView();
        }
    }

    function activateView(viewName) {
        currentView = viewName;
        tabs.forEach((tab) => {
            const active = tab.dataset.view === viewName;
            tab.classList.toggle('active', active);
            tab.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        views.forEach((view) => {
            view.classList.toggle('hidden', view.id !== 'view-' + viewName);
        });
        refreshCurrentView();
    }

    function navigateDate(delta) {
        if (!tradingDates.length) {
            return;
        }
        currentDate = resolveNearestTradingDate(currentDate);
        const index = tradingDates.indexOf(currentDate);
        const nextIndex = Math.max(0, Math.min(tradingDates.length - 1, index + delta));
        currentDate = tradingDates[nextIndex];
        renderDailyView();
    }

    function navigateWeek(delta) {
        if (!tradingDates.length) {
            return;
        }
        const weekStarts = getAvailableWeekStarts();
        const currentStart = resolveNearestWeekStart(currentWeekStart);
        const index = weekStarts.indexOf(currentStart);
        const safeIndex = index >= 0 ? index : 0;
        const nextIndex = Math.max(0, Math.min(weekStarts.length - 1, safeIndex + delta));
        currentWeekStart = weekStarts[nextIndex];
        renderWeeklyView();
    }

    function jumpToCurrentWeek() {
        if (!tradingDates.length) {
            return;
        }
        currentWeekStart = resolveNearestWeekStart(getCurrentYearAwareDefaultDate());
        renderWeeklyView();
    }

    function navigateMonth(delta) {
        if (!monthKeys.length) {
            return;
        }
        const currentIndex = monthKeys.indexOf(currentMonth);
        const safeIndex = currentIndex >= 0 ? currentIndex : 0;
        const nextIndex = Math.max(0, Math.min(monthKeys.length - 1, safeIndex + delta));
        currentMonth = monthKeys[nextIndex];
        renderMonthlyView();
    }

    function bindEvents() {
        tabs.forEach((tab) => {
            tab.addEventListener('click', () => activateView(tab.dataset.view));
        });

        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
                const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
                applyTheme(nextTheme);
                writeStoredTheme(nextTheme);
            });
        }

        document.getElementById('date-prev')?.addEventListener('click', () => navigateDate(-1));
        document.getElementById('date-next')?.addEventListener('click', () => navigateDate(1));
        document.getElementById('date-picker')?.addEventListener('change', (event) => {
            currentDate = event.target.value || currentDate;
            currentDate = resolveNearestTradingDate(currentDate);
            renderDailyView();
        });
        document.getElementById('week-prev')?.addEventListener('click', () => navigateWeek(-1));
        document.getElementById('week-next')?.addEventListener('click', () => navigateWeek(1));
        document.getElementById('week-current')?.addEventListener('click', () => jumpToCurrentWeek());
        document.getElementById('month-prev')?.addEventListener('click', () => navigateMonth(-1));
        document.getElementById('month-next')?.addEventListener('click', () => navigateMonth(1));

        document.addEventListener('mouseover', (event) => {
            const badge = getSignalBadgeTarget(event.target);
            if (badge) {
                showSeasonalTooltip(badge);
                return;
            }
            const dayBlock = getMonthDayTarget(event.target);
            if (dayBlock) {
                showMonthlyTooltip(dayBlock);
            }
        });

        document.addEventListener('mouseout', (event) => {
            const badge = getSignalBadgeTarget(event.target);
            if (badge) {
                const nextBadge = getSignalBadgeTarget(event.relatedTarget);
                if (nextBadge === badge) {
                    return;
                }
                if (activeTooltipTarget === badge) {
                    hideSeasonalTooltip();
                }
            }

            const dayBlock = getMonthDayTarget(event.target);
            if (!dayBlock) {
                return;
            }
            const nextDayBlock = getMonthDayTarget(event.relatedTarget);
            if (nextDayBlock === dayBlock) {
                return;
            }
            if (activeMonthlyTooltipTarget === dayBlock) {
                hideMonthlyTooltip();
            }
        });

        document.addEventListener('focusin', (event) => {
            const badge = getSignalBadgeTarget(event.target);
            if (badge) {
                showSeasonalTooltip(badge);
                return;
            }
            const dayBlock = getMonthDayTarget(event.target);
            if (dayBlock) {
                showMonthlyTooltip(dayBlock);
            }
        });

        document.addEventListener('focusout', (event) => {
            const badge = getSignalBadgeTarget(event.target);
            if (badge && activeTooltipTarget === badge) {
                hideSeasonalTooltip();
            }
            const dayBlock = getMonthDayTarget(event.target);
            if (dayBlock && activeMonthlyTooltipTarget === dayBlock) {
                hideMonthlyTooltip();
            }
        });

        document.addEventListener('mousemove', (event) => {
            if (activeTooltipTarget && activeTooltipTarget.contains(event.target)) {
                positionSeasonalTooltip(activeTooltipTarget);
            }
            if (activeMonthlyTooltipTarget && activeMonthlyTooltipTarget.contains(event.target)) {
                positionMonthlyTooltip(activeMonthlyTooltipTarget);
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                hideSeasonalTooltip();
                hideMonthlyTooltip();
            }
        });

        window.addEventListener('scroll', () => {
            hideSeasonalTooltip();
            hideMonthlyTooltip();
        }, true);
        window.addEventListener('resize', () => {
            if (activeTooltipTarget) {
                positionSeasonalTooltip(activeTooltipTarget);
            }
            if (activeMonthlyTooltipTarget) {
                positionMonthlyTooltip(activeMonthlyTooltipTarget);
            }
        });
    }

    initializeTheme();
    bindEvents();

    loadAlmanacFull().then((payload) => {
        if (!payload) {
            renderAlmanacError();
            return;
        }
        currentDate = getCurrentYearAwareDefaultDate();
        currentWeekStart = getWeekStart(currentDate);
        currentMonth = currentDate.slice(0, 7);
        refreshCurrentView();
    });

})();
