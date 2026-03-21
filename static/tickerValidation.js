/**
 * Ticker validation utilities — usable in browser (sets window.TickerValidation)
 * and in Node.js (module.exports) for unit testing.
 */
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = factory();       // Node.js / CommonJS
    } else {
        root.TickerValidation = factory(); // browser global
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    const _RESERVED = new Set(['TEST', 'NULL', 'NONE', 'HELP', 'NA']);

    /**
     * Instant client-side format check — no network call.
     * @param {string} input
     * @returns {{ valid: boolean, error?: string, cleaned?: string }}
     */
    function validateTickerFormat(input) {
        const cleaned = String(input == null ? '' : input).trim().toUpperCase();

        if (!cleaned) {
            return { valid: false, error: 'Please enter a stock ticker symbol.' };
        }
        if (!/^[A-Z]+$/.test(cleaned)) {
            return { valid: false, error: 'Tickers contain only letters (e.g., AAPL, MSFT).' };
        }
        if (cleaned.length > 5) {
            return { valid: false, error: 'US stock tickers are 1-5 letters long.' };
        }
        if (_RESERVED.has(cleaned)) {
            return { valid: false, error: `"${cleaned}" is not a stock ticker.` };
        }
        return { valid: true, cleaned };
    }

    /**
     * Server-side ticker verification via POST /api/validate-ticker.
     * Accepts an optional AbortSignal for external cancellation.
     * A 5-second internal timeout is always applied.
     *
     * @param {string} ticker  — already-normalised uppercase ticker
     * @param {AbortSignal|null} [externalSignal]
     * @returns {Promise<{valid: boolean, ticker?: string, company_name?: string,
     *                    error?: string, suggestions?: string[]} | null>}
     *   Returns null when cancelled via externalSignal (caller should ignore the result).
     */
    async function validateTickerRemote(ticker, externalSignal) {
        const timeoutController = new AbortController();
        const timeoutId = setTimeout(() => timeoutController.abort(), 5000);

        // Forward external cancellation to our timeout controller
        let onExternalAbort = null;
        if (externalSignal) {
            if (externalSignal.aborted) {
                clearTimeout(timeoutId);
                return null;
            }
            onExternalAbort = () => { clearTimeout(timeoutId); timeoutController.abort(); };
            externalSignal.addEventListener('abort', onExternalAbort, { once: true });
        }

        try {
            const response = await fetch('/api/validate-ticker', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker }),
                signal: timeoutController.signal,
            });
            clearTimeout(timeoutId);
            if (externalSignal && onExternalAbort) {
                externalSignal.removeEventListener('abort', onExternalAbort);
            }
            return await response.json();
        } catch (err) {
            clearTimeout(timeoutId);
            if (externalSignal && onExternalAbort) {
                externalSignal.removeEventListener('abort', onExternalAbort);
            }
            if (err.name === 'AbortError') {
                if (externalSignal && externalSignal.aborted) {
                    return null; // cancelled externally — caller should ignore
                }
                return { valid: false, error: 'Validation timed out. Please try again.' };
            }
            return { valid: false, error: 'Network error. Please check your connection.' };
        }
    }

    return { validateTickerFormat, validateTickerRemote };
}));
