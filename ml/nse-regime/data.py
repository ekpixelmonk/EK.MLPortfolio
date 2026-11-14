"""NSE market regime classifier: step 1, the data.

Downloads daily Nifty 200 and India VIX history from Yahoo Finance, caches it in data/raw_yahoo.csv
(git-ignored: Yahoo's terms do not allow redistributing their data) and prints a quality summary.
No modelling yet.

    uv run python data.py            # use the cached download, or fetch it the first time
    uv run python data.py --refresh  # download a fresh copy
"""
from __future__ import annotations

import argparse
import os
import ssl
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {"index": "^CNX200", "vix": "^INDIAVIX"}  # Nifty 200, India VIX
START_DATE = "2007-09-01"  # Yahoo's Nifty 200 history starts around here; VIX from 2008
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "raw_yahoo.csv"


def _use_system_certificates() -> None:
    """yfinance's HTTP client ignores the Windows certificate store; hand it a bundle built from it.
    Verification stays on. Does nothing if CURL_CA_BUNDLE is already set or on other platforms."""
    if os.environ.get("CURL_CA_BUNDLE") or sys.platform != "win32":
        return
    import certifi

    pems = [ssl.DER_cert_to_PEM_cert(cert) for store in ("ROOT", "CA")
            for cert, encoding, _ in ssl.enum_certificates(store) if encoding == "x509_asn"]
    bundle = Path(tempfile.gettempdir()) / "ml-portfolio-cabundle.pem"
    bundle.write_text(Path(certifi.where()).read_text() + "\n" + "\n".join(pems), encoding="utf-8")
    os.environ["CURL_CA_BUNDLE"] = str(bundle)


def download_raw() -> pd.DataFrame:
    """Daily OHLCV for the index and India VIX, columns (field, ticker). auto_adjust is off (index levels)."""
    _use_system_certificates()
    raw = yf.download(list(TICKERS.values()), start=START_DATE, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (offline, rate limited, or a certificate problem).")
    raw = raw.dropna(how="all")
    DATA_DIR.mkdir(exist_ok=True)
    raw.to_csv(RAW_CSV)
    return raw


def load_raw(refresh: bool = False) -> pd.DataFrame:
    if refresh or not RAW_CSV.exists():
        return download_raw()
    return pd.read_csv(RAW_CSV, header=[0, 1], index_col=0, parse_dates=[0])


def summarise(raw: pd.DataFrame) -> None:
    print(f"rows: {len(raw)}   {raw.index.min().date()} to {raw.index.max().date()}\n")
    for key, ticker in TICKERS.items():
        close = raw[("Close", ticker)]
        print(f"{key} ({ticker}): first close {close.first_valid_index().date()}, "
              f"last close {close.last_valid_index().date()} = {close.dropna().iloc[-1]:.2f}, "
              f"missing closes {int(close.isna().sum())}, non-positive {int((close <= 0).sum())}")
    print(f"\nduplicate dates: {int(raw.index.duplicated().sum())}")
    both = raw["Close"].dropna()
    print(f"days with both series: {len(both)}")
    print("\nlast 5 rows:")
    print(raw["Close"].tail().round(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="download fresh data instead of using the cache")
    args = parser.parse_args()
    summarise(load_raw(refresh=args.refresh))


if __name__ == "__main__":
    main()
