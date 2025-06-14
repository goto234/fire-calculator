from data_fetcher import load_market_data

if __name__ == "__main__":
    df = load_market_data(start="2000-01-01", force_reload=True, verbose=True)
    print("✅ Equity and Bond data fetched and saved to /data")
    print(df.head())
