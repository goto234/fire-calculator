import os
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
load_dotenv()  # to pull FRED_API_KEY from .env

DATA_DIR       = Path("data")
EQ_FILE        = DATA_DIR / "equity.feather"
BOND_FILE      = DATA_DIR / "bond.feather"
DIVIDEND_YIELD = 0.009                    # 0.9% annual dividend yield
FRED_SERIES_ID = "INDIRLTLT01STQ"         # OECD Long-Term Interest Rate (approx 10Y G-Sec, quarterly)

DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Feather I/O Helpers
# -----------------------------------------------------------------------------
def _read_feather(path: Path) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _write_feather(df: pd.DataFrame, path: Path):
    df_to_save = df.copy()
    df_to_save.index.name = "date"
    df_to_save.reset_index().to_feather(path)

# -------------------------------------------------------------------------
# Equity: Nifty-50 total return proxy
# -------------------------------------------------------------------------
def fetch_equity_data(
    start: str = "2000-01-01",
    verbose: bool = False,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Monthly total returns (incl. 0.9 % dividend) for ^NSEI.
    Saves/reads cache in data/equity.feather.
    """
    # 1️⃣  Serve from cache unless force_reload
    if EQ_FILE.exists() and not force_reload:
        if verbose:
            print("[Equity] Loading cached data")
        return _read_feather(EQ_FILE)

    # 2️⃣  Download from Yahoo
    if verbose:
        print(f"[Equity] Fetching ^NSEI from {start}")
    raw = yf.download(
        "^NSEI",
        start=start,
        auto_adjust=True,
        progress=verbose,
    )[["Close"]]                             # <- DataFrame (1 col)

    if raw.empty:
        raise RuntimeError("Equity download returned no data")

    monthly   = raw["Close"].resample("M").last()
    total_ret = monthly.pct_change() + DIVIDEND_YIELD / 12

    # 🔑  Ensure 1-D series (squeeze() flattens (N,1) ➜ (N,))
    if isinstance(total_ret, pd.DataFrame):
        total_ret = total_ret.iloc[:, 0]       # or .squeeze("columns")

    total_ret = total_ret.dropna()

    # wrap into DataFrame with correct index/column name
    out = total_ret.to_frame(name="equity_return")

    # Cache to feather
    _write_feather(out, EQ_FILE)
    if verbose:
        print(f"[Equity] Cached to {EQ_FILE}")
    return out


# -----------------------------------------------------------------------------
# Bond: 10-Year G-Sec Yield → Monthly Return Proxy
# -----------------------------------------------------------------------------
def fetch_bond_data(
    start: str = "2000-01-01",
    verbose: bool = False,
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Returns a monthly proxy for 10Y G-Sec returns by forward-filling
    quarterly yield observations from FRED and dividing by 12.
    Caches to data/bond.feather; reload when force_reload=True.
    """
    # 1) Use cache if present and not forcing a reload
    if BOND_FILE.exists() and not force_reload:
        if verbose:
            print("[Bond] Loading cached data")
        return _read_feather(BOND_FILE)

    # 2) Fetch quarterly yield from FRED
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set; cannot fetch bond data")

    if verbose:
        print(f"[Bond] Fetching {FRED_SERIES_ID} from FRED starting {start}")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":          FRED_SERIES_ID,
        "api_key":            api_key,
        "file_type":          "json",
        "observation_start":  start
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("observations", [])

    df = pd.DataFrame(data)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce") / 100  # decimal yield
    df = df.set_index("date")

    # 3) Forward-fill quarterly → monthly at month-start
    monthly_yield = df["value"].resample("MS").ffill()

    # 4) Proxy monthly return = (annual yield) / 12
    bond_return = monthly_yield / 12
    bond_return.index = bond_return.index.to_period("M").to_timestamp("M")

    out = bond_return.to_frame(name="bond_return").dropna()

    # 5) Cache & return
    _write_feather(out, BOND_FILE)
    if verbose:
        print(f"[Bond] Cached to {BOND_FILE}")
    return out


# -----------------------------------------------------------------------------
# Combined Market Data
# -----------------------------------------------------------------------------

def load_market_data(
    start: str = "2000-01-01",
    force_reload: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - equity_return: monthly Nifty-50 total returns
      - bond_return:   monthly G-Sec proxy returns
    """
    eq   = fetch_equity_data(start, verbose=verbose, force_reload=force_reload)
    bond = fetch_bond_data(  start, verbose=verbose, force_reload=force_reload)
    merged = eq.join(bond, how="inner")
    return merged.dropna()
