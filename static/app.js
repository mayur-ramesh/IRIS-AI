document.addEventListener('DOMContentLoaded', () => {
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

    const chartImg = document.getElementById('market-chart');
    const chartPlaceholder = document.getElementById('chart-placeholder');

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

            // Populate DOM with data
            updateDashboard(data);

            // Show Dashboard
            dashboard.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            errorMsg.textContent = error.message;
            errorMsg.classList.remove('hidden');
        } finally {
            setLoading(false);
        }
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
            headlines.forEach(headline => {
                const li = document.createElement('li');
                li.textContent = headline;
                headlinesList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = "No recent headlines found.";
            li.style.fontStyle = "italic";
            li.style.color = "var(--text-muted)";
            li.style.borderLeftColor = "transparent";
            headlinesList.appendChild(li);
        }

        // Chart display
        const chartPath = data.evidence.chart_path;
        if (chartPath) {
            // Encode path for safety, browser will fetch image and then we hide placeholder
            chartImg.src = `/api/chart?path=${encodeURIComponent(chartPath)}`;
            chartImg.onload = () => {
                chartImg.classList.remove('hidden');
                chartPlaceholder.classList.add('hidden');
            };
            chartImg.onerror = () => {
                chartImg.classList.add('hidden');
                chartPlaceholder.classList.remove('hidden');
                chartPlaceholder.textContent = "Failed to load chart image.";
            };
        } else {
            chartImg.classList.add('hidden');
            chartPlaceholder.classList.remove('hidden');
            chartPlaceholder.textContent = "Chart not available.";
        }
    }
});
