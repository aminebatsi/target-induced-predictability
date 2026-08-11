"""Data layer: daily closes from Yahoo Finance, cached as CSV.

Documented pitfall this module avoids: calling the chart API with `range=max&
interval=1d` silently DOWNSAMPLES long histories to monthly bars. We use explicit
`period1/period2` epoch bounds, which preserves true daily granularity, plus a
browser User-Agent and exponential backoff to avoid HTTP 429 rate limits.
"""
import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from config import ASSETS, DATA_DIR, ORIGINAL_ASSETS

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def yahoo_daily(ticker: str, years: int = 16, retries: int = 5) -> pd.DataFrame:
    """True DAILY bars via epoch-bounded chart API. Returns date/close/logprice."""
    sym = urllib.parse.quote(ticker)
    p2 = int(time.time()) + 86400
    p1 = p2 - int(years * 365.25 * 86400)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={p1}&period2={p2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for i in range(retries):
        try:
            res = json.loads(urllib.request.urlopen(req, timeout=30).read())["chart"]["result"][0]
            ts = np.asarray(res["timestamp"], dtype="int64")
            close = np.asarray(res["indicators"]["quote"][0]["close"], dtype=float)
            df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s").normalize(), "close": close})
            df = df.dropna()
            df = df[df["close"] > 0].drop_duplicates("date").reset_index(drop=True)
            df["logprice"] = np.log(df["close"].values)
            return df
        except Exception as exc:                        # noqa: BLE001 -- retry then raise
            last = exc
            time.sleep(3 + 3 * i)
    raise RuntimeError(f"yahoo download failed for {ticker}: {last}")


def yfinance_daily(ticker: str, years: int = 16) -> pd.DataFrame:
    """Fallback downloader via the `yfinance` package.

    The direct chart API above is preferred (no dependency, explicit epoch
    bounds), but it is rate-limited per IP and shared-cloud addresses -- Colab
    especially -- are often already throttled. yfinance handles sessions and
    retries differently and usually succeeds where the raw call does not.
    """
    import yfinance as yf

    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned nothing for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):           # single-ticker frames
        df.columns = df.columns.get_level_values(0)
    out = df.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "date", "Close": "close"}).dropna()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out[out["close"] > 0].drop_duplicates("date").reset_index(drop=True)
    out["logprice"] = np.log(out["close"].values)
    return out


def load_asset(name: str) -> pd.DataFrame:
    """Cached daily series for one asset (columns: date, close, logprice)."""
    universe = {**ASSETS, **ORIGINAL_ASSETS}
    fp = DATA_DIR / f"{name}.csv"
    if not fp.exists():
        ticker = universe[name][0]
        try:
            df = yahoo_daily(ticker)
        except Exception as exc:                        # noqa: BLE001 -- try the fallback
            print(f"  [{name}] chart API failed ({type(exc).__name__}); "
                  f"falling back to yfinance")
            df = yfinance_daily(ticker)
        df.to_csv(fp, index=False)
        time.sleep(1.0)
    return pd.read_csv(fp, parse_dates=["date"])


def download_all(verbose: bool = True) -> dict:
    """Fetch/cache the whole training universe; returns {name: n_rows} and
    sanity-checks daily granularity."""
    out = {}
    for name in ASSETS:
        df = load_asset(name)
        out[name] = len(df)
        if verbose:
            print(f"  {name:7} {len(df):5} days  {df['date'].min().date()} -> {df['date'].max().date()}")
        # granularity guard: >200 rows per year of span, else the monthly pitfall is back
        span_years = (df["date"].max() - df["date"].min()).days / 365.25
        assert len(df) > 200 * max(span_years - 1, 1), f"{name}: not daily data!"
    return out


if __name__ == "__main__":
    download_all()
