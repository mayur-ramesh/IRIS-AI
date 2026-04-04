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
    let irisLiveData = null;
    let irisPromise = null;
    let tradingDates = [];
    let monthKeys = [];

    const tabs = Array.from(document.querySelectorAll('.alm-tab'));
    const views = Array.from(document.querySelectorAll('.alm-view'));
    const themeToggle = document.getElementById('theme-toggle');

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

    async function fetchIrisLive() {
        if (irisLiveData) {
            return irisLiveData;
        }
        if (irisPromise) {
            return irisPromise;
        }
        irisPromise = fetchJson('/api/analyze?ticker=' + encodeURIComponent(SPY_TICKER))
            .then((payload) => {
                irisLiveData = payload;
                return payload;
            })
            .catch((error) => {
                console.error('Failed to load IRIS SPY data:', error);
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

    function getWeekDates(startDate) {
        if (!tradingDates.length) {
            return [];
        }
        const resolvedStart = resolveNearestTradingDate(startDate);
        const startIndex = tradingDates.findIndex((dateKey) => dateKey === resolvedStart);
        if (startIndex < 0) {
            return [];
        }
        return tradingDates.slice(startIndex, startIndex + 5);
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
        const tone = value.startsWith('RED') ? 'red' : value.startsWith('YELLOW') ? 'yellow' : 'green';
        return {
            label: value || 'UNKNOWN',
            tone,
        };
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
        if (dailyPanel) dailyPanel.innerHTML = message;
        if (weeklyTable) weeklyTable.innerHTML = message;
        if (monthlyPanel) monthlyPanel.innerHTML = message;
        if (seasonalStrip) seasonalStrip.innerHTML = message;
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
        currentWeekStart = currentDate;
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
                + '  <div class="alm-score-caption">S&amp;P 500 probability score</div>'
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

        if (irisPanel) {
            irisPanel.innerHTML = '<div class="alm-empty">Loading live IRIS SPY analysis...</div>';
        }

        const irisData = await fetchIrisLive();
        if (!irisData || !irisData.signals) {
            if (irisPanel) {
                irisPanel.innerHTML = ''
                    + '<div class="alm-empty">IRIS live SPY data is unavailable right now.'
                    + '<br><small>This comparison page reuses the existing live analysis endpoint only.</small></div>';
            }
            if (verdict) {
                verdict.innerHTML = '<span class="alm-neutral">Almanac loaded; IRIS live comparison unavailable</span>';
            }
            return;
        }

        const light = parseCheckEngineLight(irisData.signals?.check_engine_light);
        const signal = getIrisSignal(irisData);
        const trend = String(irisData.signals?.trend_label || '').trim() || 'No trend label';
        const confidence = Number(
            irisData.signals?.model_confidence
            ?? irisData.all_horizons?.['1D']?.model_confidence
        );
        const currentPrice = formatCurrency(irisData.market?.current_price);
        const predictedPrice = formatCurrency(
            irisData.market?.predicted_price_horizon
            ?? irisData.market?.predicted_price_next_session
        );
        const lightColor = light.tone === 'red'
            ? 'var(--status-red)'
            : light.tone === 'yellow'
                ? 'var(--status-yellow)'
                : 'var(--status-green)';

        if (irisPanel) {
            irisPanel.innerHTML = ''
                + '<div class="alm-score-shell">'
                + '  <div class="alm-score-caption">Check Engine Light</div>'
                + '  <div class="alm-score-value" style="color:' + lightColor + ';">' + escapeHtml(light.label.split(' ')[0]) + '</div>'
                + '</div>'
                + '<div class="alm-pill-row">' + signalBadge(signal) + '</div>'
                + '<div class="alm-stat-grid">'
                + '  <div class="alm-stat-card">'
                + '    <div class="alm-stat-label">Current Price</div>'
                + '    <div class="alm-stat-value">' + escapeHtml(currentPrice) + '</div>'
                + '  </div>'
                + '  <div class="alm-stat-card">'
                + '    <div class="alm-stat-label">Predicted Price</div>'
                + '    <div class="alm-stat-value">' + escapeHtml(predictedPrice) + '</div>'
                + '  </div>'
                + '</div>'
                + '<div class="alm-note-box">'
                + '  <strong>Trend:</strong> ' + escapeHtml(trend)
                + '  <br><strong>Confidence:</strong> ' + escapeHtml(Number.isFinite(confidence) ? confidence.toFixed(1) + '%' : 'N/A')
                + '  <br><strong>Scope:</strong> Current live SPY snapshot reused for this comparison pass.'
                + '</div>';
        }

        if (verdict) {
            const agreement = computeAgreement(signal, dayData.s_dir);
            verdict.innerHTML = ''
                + '<div class="alm-note-caption">Live IRIS vs historical Almanac direction</div>'
                + '<div style="margin-top:0.35rem; font-size:1.05rem;">' + agreementBadge(agreement) + '</div>'
                + '<div class="alm-note-caption" style="margin-top:0.4rem;">'
                + 'IRIS: ' + escapeHtml(signal)
                + ' | Almanac S&amp;P: ' + escapeHtml(dayData.s_dir) + ' (' + escapeHtml(dayData.s) + ')'
                + '</div>';
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

        const weekDates = getWeekDates(currentWeekStart);
        if (!weekDates.length) {
            if (container) container.innerHTML = '<div class="alm-empty">No trading week found for that start date.</div>';
            if (strip) strip.innerHTML = '';
            return;
        }

        currentWeekStart = weekDates[0];
        if (weekLabel) {
            weekLabel.textContent = 'Week of ' + formatShortDay(weekDates[0]) + ' - ' + formatShortDay(weekDates[weekDates.length - 1]) + ', 2026';
        }

        if (container) {
            const rows = weekDates.map((dateKey) => {
                const day = payload.daily[dateKey];
                return ''
                    + '<tr>'
                    + '  <td><strong>' + escapeHtml(day.day) + '</strong><br><span class="alm-note-caption">' + escapeHtml(dateKey) + '</span></td>'
                    + '  <td class="' + scoreClass(day.s) + '"><strong>' + escapeHtml(day.s) + '</strong></td>'
                    + '  <td>' + directionBadge(day.s_dir) + '</td>'
                    + '  <td>' + (day.icon ? '<span class="alm-chip">' + escapeHtml(day.icon.replace('_', ' ')) + '</span>' : '<span class="alm-note-caption">None</span>') + '</td>'
                    + '  <td>' + escapeHtml(day.notes || '-') + '</td>'
                    + '</tr>';
            }).join('');

            container.innerHTML = ''
                + '<table class="alm-week-table">'
                + '  <thead><tr><th>Day</th><th>S&amp;P Score</th><th>Direction</th><th>Icon</th><th>Notes</th></tr></thead>'
                + '  <tbody>' + rows + '</tbody>'
                + '</table>';
        }

        if (strip) {
            strip.innerHTML = weekDates.map((dateKey) => {
                const day = payload.daily[dateKey];
                return ''
                    + '<div class="alm-agree-cell">'
                    + '  <div class="alm-weekday">' + escapeHtml(day.day) + '</div>'
                    + '  <div class="alm-mini-value ' + scoreClass(day.s) + '">' + escapeHtml(day.s_dir) + '</div>'
                    + '  <div class="alm-weekday">' + escapeHtml(day.s) + ' S&amp;P</div>'
                    + '</div>';
            }).join('');
        }
    }

    async function renderMonthlyView() {
        const payload = await loadAlmanacFull();
        const monthLabel = document.getElementById('month-label');
        const almanacPanel = document.getElementById('almanac-monthly-content');
        const irisPanel = document.getElementById('iris-monthly-content');
        const calendar = document.getElementById('monthly-calendar-heatmap');

        if (!payload) {
            if (almanacPanel) almanacPanel.innerHTML = '<div class="alm-empty">Monthly Almanac data could not be loaded.</div>';
            if (irisPanel) irisPanel.innerHTML = '<div class="alm-empty">Monthly IRIS summary unavailable.</div>';
            if (calendar) calendar.innerHTML = '<div class="alm-empty">Calendar unavailable.</div>';
            return;
        }

        if (!monthKeys.includes(currentMonth)) {
            currentMonth = monthKeys.includes('2026-04') ? '2026-04' : monthKeys[0];
        }

        const monthInfo = payload.months[currentMonth];
        if (!monthInfo) {
            if (almanacPanel) almanacPanel.innerHTML = '<div class="alm-empty">No monthly data found.</div>';
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
                + 'Monthly IRIS aggregation is intentionally lightweight in this pass.'
                + '<br><br>The page currently reuses live SPY analysis for daily comparison only.'
                + '<br><br>Planned future extension: stored historical IRIS reports by session date for weekly and monthly accuracy summaries.'
                + '</div>';
        }

        if (calendar) {
            const [yearString, monthString] = currentMonth.split('-');
            const year = Number(yearString);
            const monthNumber = Number(monthString);
            const daysInMonth = new Date(year, monthNumber, 0).getDate();
            const monthStart = new Date(currentMonth + '-01T12:00:00');
            const mondayOffset = (monthStart.getDay() + 6) % 7;
            const pieces = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((label) => {
                return '<div class="alm-weekday" style="text-align:center; padding:0.2rem;">' + label + '</div>';
            });

            for (let index = 0; index < mondayOffset; index += 1) {
                pieces.push('<div class="alm-cal-day empty"></div>');
            }

            for (let day = 1; day <= daysInMonth; day += 1) {
                const dateKey = currentMonth + '-' + String(day).padStart(2, '0');
                const dayData = payload.daily[dateKey];
                if (dayData) {
                    const background = Number(dayData.s) >= 60
                        ? 'rgba(63, 127, 107, 0.48)'
                        : Number(dayData.s) <= 40
                            ? 'rgba(182, 78, 90, 0.48)'
                            : 'rgba(180, 140, 58, 0.3)';
                    pieces.push(
                        '<div class="alm-cal-day" style="background:' + background + '; border-color: var(--panel-border);" title="'
                        + escapeHtml(dateKey + ': S&P ' + dayData.s + ' (' + dayData.s_dir + ')')
                        + '">'
                        + '  <div class="alm-cal-daynum">' + day + '</div>'
                        + '  <div class="alm-calendar-score">' + escapeHtml(dayData.s) + '</div>'
                        + '</div>'
                    );
                } else {
                    const currentDateObj = new Date(dateKey + 'T12:00:00');
                    const dayOfWeek = currentDateObj.getDay();
                    const weekend = dayOfWeek === 0 || dayOfWeek === 6;
                    pieces.push(
                        '<div class="alm-cal-day' + (weekend ? ' empty' : '') + '"'
                        + (weekend ? '' : ' style="background: var(--bg-subtle); border-color: var(--panel-border);"')
                        + '>'
                        + '  <div class="alm-cal-daynum">' + day + '</div>'
                        + (weekend ? '' : '  <div class="alm-calendar-score">Holiday</div>')
                        + '</div>'
                    );
                }
            }

            calendar.innerHTML = pieces.join('');
        }
    }

    async function renderSeasonalView() {
        const payload = await loadAlmanacFull();
        const strip = document.getElementById('seasonal-heatmap-strip');
        const signalsList = document.getElementById('seasonal-signals-list');
        const ranking = document.getElementById('seasonal-monthly-ranking');

        if (!payload) {
            if (strip) strip.innerHTML = '<div class="alm-empty">Seasonal data could not be loaded.</div>';
            if (signalsList) signalsList.innerHTML = '';
            if (ranking) ranking.innerHTML = '';
            return;
        }

        if (strip) {
            strip.innerHTML = monthKeys.map((monthKey) => {
                const monthInfo = payload.months[monthKey];
                const heatmapEntry = payload.heatmap[monthKey] || {};
                const rankValue = getMonthDisplayRank(monthKey, monthInfo, heatmapEntry);
                const isCurrent = monthKey === currentMonth;
                const monthLabel = new Date(monthKey + '-01T12:00:00').toLocaleDateString('en-US', { month: 'short' });
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

        if (signalsList) {
            const signals = Array.isArray(payload.signals) ? payload.signals : [];
            if (!signals.length) {
                signalsList.innerHTML = '<div class="alm-note-box" style="width:100%;">No tagged seasonal signals were extracted for this dataset.</div>';
            } else {
                signalsList.innerHTML = signals.map((signal) => {
                    const typeClass = signal.type === 'timing'
                        ? 'alm-signal-timing'
                        : signal.type === 'weekly_timing'
                            ? 'alm-signal-weekly'
                            : signal.type === 'risk_indicator'
                                ? 'alm-signal-risk'
                                : 'alm-signal-macro';
                    const title = signal.label + ' (' + signal.source_month + '): ' + signal.description;
                    return '<div class="alm-signal-badge ' + typeClass + '" title="' + escapeHtml(title) + '">' + escapeHtml(signal.label) + '</div>';
                }).join('');
            }
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
    }

    function refreshCurrentView() {
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
        const weekDates = getWeekDates(currentWeekStart);
        const currentStart = weekDates[0] || resolveNearestTradingDate(currentWeekStart);
        const index = tradingDates.indexOf(currentStart);
        const nextIndex = Math.max(0, Math.min(tradingDates.length - 5, index + (delta * 5)));
        currentWeekStart = tradingDates[nextIndex];
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
        document.getElementById('month-prev')?.addEventListener('click', () => navigateMonth(-1));
        document.getElementById('month-next')?.addEventListener('click', () => navigateMonth(1));
    }

    initializeTheme();
    bindEvents();

    loadAlmanacFull().then((payload) => {
        if (!payload) {
            renderAlmanacError();
            return;
        }
        currentDate = getCurrentYearAwareDefaultDate();
        currentWeekStart = currentDate;
        currentMonth = currentDate.slice(0, 7);
        refreshCurrentView();
    });

})();
