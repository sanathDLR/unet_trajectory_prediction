# utils/timestamp_utils.py
import re, pandas as pd
_RE = re.compile(r"(\d{8}_\d{9})")  # YYYYmmdd_HHMMSSmmm

def extract_ts_token(name: str) -> str:
    m = _RE.search(name)
    return m.group(1) if m else name

def format_ts_token(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d_%H%M%S%f")[:-3]

def parse_ts_token(token: str) -> pd.Timestamp:
    """
    Convert 'YYYYmmdd_HHMMSSmmm' token back to pandas.Timestamp.
    """
    return pd.to_datetime(token, format="%Y%m%d_%H%M%S%f")