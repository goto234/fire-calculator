import os
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
load_dotenv()

DATA_DIR = Path("data")
DIVIDEND_YIELD = 0.009  # 0.9% annual dividend yield

DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Feather I/O Helpers
# -----------------------------------------------------------------------------
def _read_feather(path: Path) -> pd.DataFrame:
    """Read feather file and set date index."""
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _write_feather(df: pd.DataFrame, path: Path):
    """Write DataFrame to feather with date index reset."""
    df_to_save = df.copy()
    df_to_save.index.name = "date"
    df_to_save.reset_index().to_feather(path)


# -----------------------------------------------------------------------------
# Equity Data Fetcher
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_equity_data(
    ticker: str = "^NSEI",
    start: str = "2000-01-01",
    return_window: int = 15,
    verbose: bool = False,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Fetch equity data with monthly total returns including dividends.
    Uses return_window to limit data to recent N years.
    """
    cache_file = DATA_DIR / f"equity_{ticker.replace('^', '')}.feather"
    
    # Calculate actual start date based on return window
    end_date = pd.Timestamp.now()
    actual_start = end_date - pd.DateOffset(years=return_window)
    actual_start_str = actual_start.strftime('%Y-%m-%d')
    
    # Use cache if exists and not forcing reload
    if cache_file.exists() and not force_reload:
        try:
            cached_data = _read_feather(cache_file)
            # Filter to return window
            filtered_data = cached_data[cached_data.index >= actual_start]
            if not filtered_data.empty and verbose:
                st.info(f"[Equity] Using cached data for {ticker}")
            if not filtered_data.empty:
                return filtered_data
        except Exception as e:
            if verbose:
                st.warning(f"Cache read failed: {e}")

    try:
        if verbose:
            st.info(f"[Equity] Fetching {ticker} from {actual_start_str}")
        
        # Download from Yahoo Finance
        raw = yf.download(
            ticker,
            start=max(start, actual_start_str),
            auto_adjust=True,
            progress=False,
        )[["Close"]]

        if raw.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        # Calculate monthly returns with dividends
        monthly_prices = raw["Close"].resample("M").last()
        monthly_returns = monthly_prices.pct_change() + DIVIDEND_YIELD / 12
        
        # Ensure it's a Series
        if isinstance(monthly_returns, pd.DataFrame):
            monthly_returns = monthly_returns.iloc[:, 0]
        
        monthly_returns = monthly_returns.dropna()
        
        # Convert to DataFrame
        result = monthly_returns.to_frame(name="equity_return")
        
        # Cache the full result
        _write_feather(result, cache_file)
        
        # Return filtered data
        return result[result.index >= actual_start]
        
    except Exception as e:
        if verbose:
            st.error(f"Failed to fetch {ticker}: {e}")
        # Try to return cached data if available
        if cache_file.exists():
            try:
                cached_data = _read_feather(cache_file)
                return cached_data[cached_data.index >= actual_start]
            except:
                pass
        return pd.DataFrame(columns=["equity_return"])


# -----------------------------------------------------------------------------
# Bond Data Fetcher
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_bond_data(
    series_id: str = "INDIRLTLT01STQ",
    start: str = "2000-01-01",
    return_window: int = 15,
    verbose: bool = False,
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Fetch bond yield data from FRED and convert to monthly returns.
    Uses return_window to limit data to recent N years.
    """
    cache_file = DATA_DIR / f"bond_{series_id}.feather"
    
    # Calculate actual start date based on return window
    end_date = pd.Timestamp.now()
    actual_start = end_date - pd.DateOffset(years=return_window)
    actual_start_str = actual_start.strftime('%Y-%m-%d')
    
    # Use cache if exists and not forcing reload
    if cache_file.exists() and not force_reload:
        try:
            cached_data = _read_feather(cache_file)
            filtered_data = cached_data[cached_data.index >= actual_start]
            if not filtered_data.empty and verbose:
                st.info(f"[Bond] Using cached data for {series_id}")
            if not filtered_data.empty:
                return filtered_data
        except Exception as e:
            if verbose:
                st.warning(f"Cache read failed: {e}")

    try:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY not set in environment")

        if verbose:
            st.info(f"[Bond] Fetching {series_id} from FRED")
        
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": max(start, actual_start_str)
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("observations", [])
        
        if not data:
            raise RuntimeError(f"No data returned for {series_id}")

        # Process bond data
        df = pd.DataFrame(data)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce") / 100  # Convert to decimal
        df = df.set_index("date").dropna()

        # Forward fill quarterly data to monthly and convert to returns
        monthly_yield = df["value"].resample("MS").ffill()
        bond_return = monthly_yield / 12  # Simple monthly return approximation
        
        # Align to month-end
        bond_return.index = bond_return.index.to_period("M").to_timestamp("M")
        
        result = bond_return.to_frame(name="bond_return").dropna()
        
        # Cache the full result
        _write_feather(result, cache_file)
        
        # Return filtered data
        return result[result.index >= actual_start]
        
    except Exception as e:
        if verbose:
            st.error(f"Failed to fetch {series_id}: {e}")
        # Try to return cached data if available
        if cache_file.exists():
            try:
                cached_data = _read_feather(cache_file)
                return cached_data[cached_data.index >= actual_start]
            except:
                pass
        return pd.DataFrame(columns=["bond_return"])


# -----------------------------------------------------------------------------
# Combined Market Data Loader
# -----------------------------------------------------------------------------
def load_market_data(
    equity_ticker: str = "^NSEI",
    bond_series: str = "INDIRLTLT01STQ",
    return_window: int = 15,
    start: str = "2000-01-01",
    force_reload: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load and combine equity and bond data.
    Returns DataFrame with equity_return and bond_return columns.
    """
    equity_data = fetch_equity_data(
        ticker=equity_ticker,
        start=start,
        return_window=return_window,
        verbose=verbose,
        force_reload=force_reload
    )
    
    bond_data = fetch_bond_data(
        series_id=bond_series,
        start=start,
        return_window=return_window,
        verbose=verbose,
        force_reload=force_reload
    )
    
    # Combine data
    if equity_data.empty or bond_data.empty:
        if verbose:
            st.warning("Missing equity or bond data, using fallback synthetic data")
        return _generate_fallback_data(return_window)
    
    combined = equity_data.join(bond_data, how="inner")
    return combined.dropna()


def _generate_fallback_data(return_window: int) -> pd.DataFrame:
    """Generate synthetic market data as fallback."""
    import numpy as np
    
    # Generate monthly data for the return window
    dates = pd.date_range(
        end=pd.Timestamp.now().replace(day=1) - pd.DateOffset(days=1),
        periods=return_window * 12,
        freq="ME"
    )
    
    np.random.seed(42)  # Reproducible
    
    # Synthetic equity returns (higher volatility)
    equity_returns = np.random.normal(0.008, 0.04, len(dates))  # ~10% annual, 15% vol
    
    # Synthetic bond returns (lower volatility)
    bond_returns = np.random.normal(0.005, 0.015, len(dates))  # ~6% annual, 5% vol
    
    return pd.DataFrame({
        "equity_return": equity_returns,
        "bond_return": bond_returns
    }, index=dates)


# -----------------------------------------------------------------------------
# Preset Scenarios
# -----------------------------------------------------------------------------
def get_preset_scenarios():
    """Return predefined market stress scenarios."""
    return {
        "2008 Crash": {"equity_stress": -0.45, "bond_stress": 0.05},
        "Covid-March": {"equity_stress": -0.30, "bond_stress": 0.02},
        "Dot-Com": {"equity_stress": -0.50, "bond_stress": 0.00},
    }