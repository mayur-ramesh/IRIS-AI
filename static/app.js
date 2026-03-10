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
        '1M': { period: '1mo', interval: '1h' },
        '6M': { period: '6mo', interval: '1d' },
        'YTD': { period: 'ytd', interval: '1d' },
        '1Y': { period: '1y', interval: '1d' },
        '5Y': { period: '5y', interval: '1wk' },
        'MAX': { period: 'max', interval: '1mo' },
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

    const chartContainer = document.getElementById('advanced-chart');
    const chartPlaceholder = document.getElementById('chart-placeholder');
    let lwChart = null;
    let currentTicker = '';
    const usdFormatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });

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

    async function loadTickerData(ticker, timeframeKey, keepDashboardVisible = false) {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        if (!normalizedTicker) return;

        const activeTimeframe = TIMEFRAME_TO_QUERY[timeframeKey] ? timeframeKey : '6M';
        const mapped = TIMEFRAME_TO_QUERY[activeTimeframe];
        const params = new URLSearchParams({
            ticker: normalizedTicker,
            timeframe: activeTimeframe,
            period: mapped.period,
            interval: mapped.interval,
        });

        setLoading(true);
        errorMsg.classList.add('hidden');
        if (!keepDashboardVisible) {
            dashboard.classList.add('hidden');
        }

        try {
            const response = await fetch(`/api/analyze?${params.toString()}`);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch data');
            }

            dashboard.classList.remove('hidden');
            currentTicker = String(data?.meta?.symbol || normalizedTicker).toUpperCase();
            input.value = currentTicker;
            updateDashboard(data);
        } catch (error) {
            console.error(error);
            errorMsg.textContent = error.message || 'Failed to fetch data';
            errorMsg.classList.remove('hidden');
            if (!keepDashboardVisible) {
                dashboard.classList.add('hidden');
            }
        } finally {
            setLoading(false);
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const ticker = input.value.trim().toUpperCase();
        if (!ticker) return;
        await loadTickerData(ticker, getActiveTimeframe(), false);
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
            await loadTickerData(ticker, timeframeKey, true);
        });
    });
    setActiveTimeframe(getActiveTimeframe());

    window.addEventListener('resize', () => {
        resizeChartToContainer();
    });

    function setLoading(isLoading) {
        if (isLoading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            timeframeButtons.forEach((btn) => { btn.disabled = true; });
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
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
            normalized.push({ time: normalizedTime, value: rawValue });
        });
        return normalized;
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
            hour12: false,
        }).format(dt);
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

        // Headlines
        headlinesList.innerHTML = '';
        const headlines = data.evidence.headlines_used;
        if (headlines && headlines.length > 0) {
            headlines.forEach((headline) => {
                let title = '';
                let url = '';
                if (headline && typeof headline === 'object') {
                    title = String(headline.title || '').trim();
                    url = String(headline.url || '').trim();
                } else {
                    title = String(headline || '').trim();
                }
                if (!title) return;

                const li = document.createElement('li');
                const isHttpUrl = /^https?:\/\//i.test(url);
                if (isHttpUrl) {
                    const link = document.createElement('a');
                    link.href = url;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.className = 'headline-link';
                    link.textContent = title;
                    li.appendChild(link);
                } else {
                    li.textContent = title;
                }
                headlinesList.appendChild(li);
            });
            if (!headlinesList.children.length) {
                const li = document.createElement('li');
                li.textContent = "No recent headlines found.";
                li.style.fontStyle = "italic";
                li.style.color = "var(--text-muted)";
                li.style.borderLeftColor = "transparent";
                headlinesList.appendChild(li);
            }
        } else {
            const li = document.createElement('li');
            li.textContent = "No recent headlines found.";
            li.style.fontStyle = "italic";
            li.style.color = "var(--text-muted)";
            li.style.borderLeftColor = "transparent";
            headlinesList.appendChild(li);
        }

        // Chart display
        if (lwChart) {
            lwChart.remove();
            lwChart = null;
        }
        chartTooltip.style.display = 'none';

        const history = normalizeHistoryPoints(data?.market?.history);
        if (history.length > 0) {
            chartPlaceholder.classList.add('hidden');
            const { width: chartWidth, height: chartHeight } = getChartDimensions();

            const isUptrend = data.signals.trend_label.includes('UPTREND');
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

            let areaSeries;
            if (typeof lwChart.addAreaSeries === 'function') {
                areaSeries = lwChart.addAreaSeries({
                    lineColor: lineColor,
                    topColor: topColor,
                    bottomColor: bottomColor,
                    lineWidth: 2,
                    priceFormat: {
                        type: 'price',
                        precision: 2,
                        minMove: 0.01,
                    },
                });
            } else {
                // Version 5+ syntax
                areaSeries = lwChart.addSeries(LightweightCharts.AreaSeries, {
                    lineColor: lineColor,
                    topColor: topColor,
                    bottomColor: bottomColor,
                    lineWidth: 2,
                    priceFormat: {
                        type: 'price',
                        precision: 2,
                        minMove: 0.01,
                    },
                });
            }

            areaSeries.setData(history);
            lwChart.subscribeCrosshairMove((param) => {
                if (!param || !param.point || !param.time) {
                    chartTooltip.style.display = 'none';
                    return;
                }
                if (!chartContainer || param.point.x < 0 || param.point.y < 0) {
                    chartTooltip.style.display = 'none';
                    return;
                }

                const priceAtCursor = readCrosshairPrice(param, areaSeries);
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

            // Predict line
            const lastDataPoint = history[history.length - 1];
            if (data.market.predicted_price_next_session) {
                let predTime = null;
                const lastTime = lastDataPoint.time;
                const intervalHint = String((data.meta && data.meta.interval) || '').toLowerCase();

                if (typeof lastTime === 'number' && Number.isFinite(lastTime)) {
                    let stepSeconds = 24 * 60 * 60; // default daily step
                    if (history.length >= 2) {
                        const prevTime = history[history.length - 2].time;
                        if (typeof prevTime === 'number' && Number.isFinite(prevTime) && lastTime > prevTime) {
                            stepSeconds = Math.max(60, Math.round(lastTime - prevTime));
                        }
                    } else if (intervalHint.endsWith('h')) {
                        const hours = parseInt(intervalHint, 10);
                        if (Number.isFinite(hours) && hours > 0) {
                            stepSeconds = hours * 60 * 60;
                        }
                    } else if (intervalHint.endsWith('m')) {
                        const mins = parseInt(intervalHint, 10);
                        if (Number.isFinite(mins) && mins > 0) {
                            stepSeconds = mins * 60;
                        }
                    }
                    predTime = lastTime + stepSeconds;
                } else {
                    const predictedDate = new Date(lastTime);
                    predictedDate.setDate(predictedDate.getDate() + 1);

                    // skip weekend for legacy date-string datasets
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
                    { time: predTime, value: data.market.predicted_price_next_session }
                ]);

                // Create a marker for the prediction
                lineSeries.setMarkers([
                    {
                        time: predTime,
                        position: 'aboveBar',
                        color: '#f59e0b',
                        shape: 'circle',
                        text: 'Predicted',
                    }
                ]);
            }

            lwChart.timeScale().fitContent();

        } else {
            if (chartContainer) {
                chartPlaceholder.classList.remove('hidden');
                chartPlaceholder.textContent = "Chart not available.";
                chartTooltip.style.display = 'none';
            }
        }
    }
});
