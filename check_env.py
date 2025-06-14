# check_env.py
import os
from dotenv import load_dotenv

load_dotenv()
fred_key = os.getenv("FRED_API_KEY")

if fred_key:
    print("FRED key =", fred_key[:6] + "…")
else:
    print("FRED_API_KEY not found")
