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
    const themeToggle = document.getElementById('theme-toggle');
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

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const ticker = input.value.trim().toUpperCase();
        if (!ticker) return;

        // UI Loading State
        setLoading(true);
        errorMsg.classList.add('hidden');
        dashboard.classList.add('hidden'); // Hide old results

        try {
            const response = await fetch(`/api/analyze?ticker=${ticker}`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch data');
            }

            // Show Dashboard first so containers have dimensions
            dashboard.classList.remove('hidden');

            // Populate DOM with data (including charts which depend on clientWidth)
            updateDashboard(data);

        } catch (error) {
            console.error(error);
            errorMsg.textContent = error.message;
            errorMsg.classList.remove('hidden');
            dashboard.classList.add('hidden');
        } finally {
            setLoading(false);
        }
    });

    window.addEventListener('resize', () => {
        resizeChartToContainer();
    });

    function setLoading(isLoading) {
        if (isLoading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    }

    function updateDashboard(data) {
        // Meta
        resTicker.textContent = data.meta.symbol;
        const date = new Date(data.meta.generated_at);
        resTime.textContent = `Updated: ${date.toLocaleString()} (${data.meta.mode.toUpperCase()} MODE)`;

        // Prices
        const currencyFormatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
        currentPriceEl.textContent = currencyFormatter.format(data.market.current_price);
        predictedPriceEl.textContent = currencyFormatter.format(data.market.predicted_price_next_session);

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

        const history = data.market.history;
        if (history && history.length > 0) {
            chartPlaceholder.classList.add('hidden');
            const { width: chartWidth, height: chartHeight } = getChartDimensions();

            const isUptrend = data.signals.trend_label.includes('UPTREND');
            const lineColor = isUptrend ? '#10b981' : '#ef4444';
            const topColor = isUptrend ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)';
            const bottomColor = isUptrend ? 'rgba(16, 185, 129, 0.0)' : 'rgba(239, 68, 68, 0.0)';

            const cssVars = window.getComputedStyle(document.documentElement);
            const chartTextColor = cssVars.getPropertyValue('--text-muted').trim() || '#9ca3af';
            const chartGridColor = cssVars.getPropertyValue('--chart-border').trim() || 'rgba(255, 255, 255, 0.08)';

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
                    rightOffset: 2,
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
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

            // Predict line
            const lastDataPoint = history[history.length - 1];
            if (data.market.predicted_price_next_session) {
                const predictedDate = new Date(lastDataPoint.time);
                predictedDate.setDate(predictedDate.getDate() + 1);

                // skip weekend
                if (predictedDate.getDay() === 6) predictedDate.setDate(predictedDate.getDate() + 2);
                if (predictedDate.getDay() === 0) predictedDate.setDate(predictedDate.getDate() + 1);

                const y = predictedDate.getFullYear();
                const m = String(predictedDate.getMonth() + 1).padStart(2, '0');
                const d = String(predictedDate.getDate()).padStart(2, '0');
                const predTime = `${y}-${m}-${d}`;

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
            }
        }
    }
});
