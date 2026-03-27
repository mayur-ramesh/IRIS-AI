/**
 * Ticker validation utilities — usable in browser (sets window.TickerValidation)
 * and in Node.js (module.exports) for unit testing.
 *
 * All rejection objects carry a ``code`` field for structured error handling.
 */
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = factory();       // Node.js / CommonJS
    } else {
        root.TickerValidation = factory(); // browser global
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    // -----------------------------------------------------------------------
    // Error codes (mirror the Python ErrorCode class)
    // -----------------------------------------------------------------------
    const ErrorCodes = {
        EMPTY_INPUT:       'EMPTY_INPUT',
        INVALID_FORMAT:    'INVALID_FORMAT',
        RESERVED_WORD:     'RESERVED_WORD',
        TICKER_NOT_FOUND:  'TICKER_NOT_FOUND',
        TICKER_DELISTED:   'TICKER_DELISTED',
        API_TIMEOUT:       'API_TIMEOUT',
        API_ERROR:         'API_ERROR',
        RATE_LIMITED:      'RATE_LIMITED',
        DATA_FETCH_FAILED: 'DATA_FETCH_FAILED',
        INTERNAL_ERROR:    'INTERNAL_ERROR',
    };

    // -----------------------------------------------------------------------
    // Known non-stock inputs (must be uppercase)
    // -----------------------------------------------------------------------
    const _RESERVED = new Set(['TEST', 'NULL', 'NONE', 'HELP', 'NA']);

    const _CRYPTO = new Set([
        'BTC', 'ETH', 'XRP', 'LTC', 'BNB', 'SOL', 'ADA', 'DOT',
        'AVAX', 'DOGE', 'MATIC', 'SHIB', 'TRX', 'LINK', 'ATOM', 'USDT', 'USDC',
    ]);

    const _CRYPTO_MESSAGE =
        'IRIS-AI analyzes stocks and ETFs. ' +
        'For cryptocurrency analysis, please use a crypto-specific platform.';

    const _MAX_RAW_LENGTH = 20;   // chars before any processing

    // -----------------------------------------------------------------------
    // Layer 0 – input sanitisation
    // -----------------------------------------------------------------------

    /**
     * Clean arbitrary user input into a normalised ticker string.
     * Mirrors the Python ``sanitize_ticker_input`` function.
     *
     * @param {string} raw
     * @returns {string} uppercase, cleaned ticker (may still be invalid format)
     */
    function sanitizeTicker(raw) {
        let s = String(raw == null ? '' : raw).trim();
        if (s.length > _MAX_RAW_LENGTH) s = s.slice(0, _MAX_RAW_LENGTH);
        s = s.replace(/^[\$#]+/, '');                           // $ / # prefix
        s = s.replace(/^ticker:/i, '');                         // "ticker:" prefix
        s = s.replace(/\s+(stock|etf|shares)$/i, '');          // trailing words
        s = s.replace(/\s+/g, '');                             // internal spaces
        return s.toUpperCase();
    }

    // -----------------------------------------------------------------------
    // Layer 1 – format check (client-side, instant)
    // -----------------------------------------------------------------------

    // 1-5 letters optionally followed by ONE dot and 1-2 letters (e.g. BRK.B)
    const _TICKER_RE = /^[A-Z]{1,5}(\.[A-Z]{1,2})?$/;

    // Yahoo special symbols (mirrors Python ticker_validator.py)
    const _INDEX_RE     = /^\^[A-Z0-9.\-]{1,14}$/;                       // ^GSPC, ^DJI, ^IXIC
    const _FUTURES_RE   = /^[A-Z0-9]{1,8}=F$/;                           // CL=F, GC=F, SI=F
    const _COMPOSITE_RE = /^[A-Z0-9]{1,8}-[A-Z0-9]{1,8}\.[A-Z]{1,6}$/;  // DX-Y.NYB

    function _isSpecialMarketSymbol(ticker) {
        return _INDEX_RE.test(ticker) || _FUTURES_RE.test(ticker) || _COMPOSITE_RE.test(ticker);
    }

    /**
     * Instant client-side format check — no network call.
     * Sanitises *input* before checking.
     *
     * @param {string} input
     * @returns {{ valid: boolean, code?: string, error?: string, cleaned?: string }}
     */
    function validateTickerFormat(input) {
        const cleaned = sanitizeTicker(input);

        if (!cleaned) {
            return { valid: false, code: ErrorCodes.EMPTY_INPUT,
                     error: 'Please enter a stock ticker symbol.' };
        }
        if (_CRYPTO.has(cleaned)) {
            return { valid: false, code: ErrorCodes.RESERVED_WORD,
                     error: _CRYPTO_MESSAGE };
        }
        if (!_TICKER_RE.test(cleaned) && !_isSpecialMarketSymbol(cleaned)) {
            return { valid: false, code: ErrorCodes.INVALID_FORMAT,
                     error: 'Invalid ticker format. Use stock format (e.g., AAPL, BRK.B) or special market symbols (e.g., ^GSPC, CL=F).' };
        }
        if (_RESERVED.has(cleaned)) {
            return { valid: false, code: ErrorCodes.RESERVED_WORD,
                     error: `"${cleaned}" is not a stock ticker.` };
        }
        return { valid: true, cleaned };
    }

    // -----------------------------------------------------------------------
    // Layer 2 – server-side verification
    // -----------------------------------------------------------------------

    /**
     * Server-side ticker verification via POST /api/validate-ticker.
     * Accepts an optional AbortSignal for external cancellation.
     * A 5-second internal timeout is always applied.
     *
     * @param {string} ticker  — sanitised uppercase ticker
     * @param {AbortSignal|null} [externalSignal]
     * @returns {Promise<object|null>}
     *   Returns null when cancelled via externalSignal (caller should ignore).
     */
    async function validateTickerRemote(ticker, externalSignal) {
        const timeoutController = new AbortController();
        const timeoutId = setTimeout(() => timeoutController.abort(), 5000);

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
                    return null;   // cancelled externally — caller ignores
                }
                return { valid: false, code: ErrorCodes.API_TIMEOUT,
                         error: 'Validation timed out. Please try again.' };
            }
            return { valid: false, code: ErrorCodes.API_ERROR,
                     error: 'Network error. Please check your connection.' };
        }
    }

    return { validateTickerFormat, validateTickerRemote, sanitizeTicker, ErrorCodes };
}));
