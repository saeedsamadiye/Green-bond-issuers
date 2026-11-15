"""
greenbond_engine.py

Do investors gain from holding the shares of companies that issue green bonds?

The design follows Badía, Cortez and Silva (2024, Economics Letters): build a
calendar-time portfolio of green bond issuers, build the same portfolio from
matched non-issuers, and compare the two with Fama-French factor models. An
event study of the days around each issuer's first green bond is added.

Version 2 changes, compared with the first version of the project:
  * every share price is converted to US dollars before returns are computed,
    because the Kenneth French factors and risk-free rate are in US dollars;
  * control firms are matched in the same sector first (same country, then
    region, then anywhere), then in the same country or region, and only then
    in the whole pool (the pool now includes the FTSE 100 and the SMI);
  * the event study adds twin-adjusted abnormal returns, a pre-listing window,
    an optional earnings-date filter and optional announcement dates.

Data (all free):
  Euronext ESG bond list           bonds and issuers
  Yahoo Finance (yfinance)         daily share prices, firm country, earnings dates
  Kenneth French data library      European five factors, monthly and daily
  European Central Bank            daily euro reference exchange rates

Sections
  1  Settings
  2  The bond list
  3  Finding share tickers
  4  Market data: prices, factors, exchange rates, firm info, earnings dates
  5  Returns, currency conversion and control firms
  6  Calendar-time portfolios and factor regressions
  7  Event study
  8  Full run, comparison of designs, robustness grid
  9  Demo data
  10 Saved results
  11 Command line

Author: Saeed Samadiye
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =============================================================================
# 1  Settings
# =============================================================================

FACTOR_MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}

# the four types in the Euronext list (it has no blue or transition bonds)
BOND_TYPES = ("Green", "Sustainability", "Social", "Sustainability-Linked")


@dataclass
class Settings:
    """Every choice the results depend on. The dashboard edits these."""

    # sample
    start: str = "2014-01-01"
    end: str = "2025-12-31"
    bond_types: tuple = ("Green",)
    min_nominal_mm: float = 0.0
    exclude_public_issuers: bool = True

    # currency: "USD" converts all prices to dollars, like the French factors.
    # "local" keeps each share in its own currency (version 1 behaviour).
    currency: str = "USD"

    # portfolios
    hold_months: int = 36
    entry_lag_months: int = 1
    min_firms: int = 5
    winsor_pct: float = 1.0

    # matching
    # "country_region": same country if a close enough twin exists, then same
    # region, then anyone. "sector_country": same sector first (with country,
    # then region, then anywhere), then as country_region. "pool": nearest
    # firm in the whole pool (version 1).
    match_mode: str = "sector_country"
    match_caliper: float = 0.25          # max standardised distance accepted in a country or region
    match_window_months: int = 36
    match_min_months: int = 18
    match_with_replacement: bool = False

    # regressions
    factor_model: str = "FF5"
    nw_lags: int = 6

    # event study (trading days relative to the event date)
    event_date_source: str = "listing"   # or "announcement" (uses data/announcement_dates.csv when filled)
    est_window: tuple = (-250, -30)
    event_windows: tuple = ((-10, -2), (-1, 1), (0, 5), (-5, 5))
    event_min_est_days: int = 120
    exclude_earnings: bool = True        # only has an effect when earnings dates are available
    earnings_window: tuple = (-5, 5)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def settings_from_dict(d: dict) -> Settings:
    """Rebuild Settings from JSON, where tuples come back as lists."""
    d = dict(d)
    known = Settings.__dataclass_fields__
    d = {k: v for k, v in d.items() if k in known}
    for k in ("bond_types", "est_window", "earnings_window"):
        if k in d:
            d[k] = tuple(d[k])
    if "event_windows" in d:
        d["event_windows"] = tuple(tuple(w) for w in d["event_windows"])
    return Settings(**d)


# =============================================================================
# 2  The bond list
# =============================================================================

_COLUMN_ALIASES = {
    "issuer": ["issuer", "issuer name", "emetteur", "émetteur"],
    "isin": ["isin", "isin code"],
    "listing_date": ["listing date", "date of listing", "issue date", "first listing date"],
    "nominal_mm": ["amount (million)", "nominal (mm)", "nom. (mm)", "nominal", "amount (mm)",
                   "issued amount (mm)", "nominal amount"],
    "currency": ["currency", "ccy"],
    "type": ["type", "bond type", "esg type", "label"],
    "venue": ["listing venue", "venue", "market", "location"],
}

# Names that mark a government, public body, supranational or securitisation
# vehicle: none of them has a share to study.
_PUBLIC_PATTERNS = [
    # states
    r"republic", r"repubblica", r"kingdom", r"regno", r"state of", r"etat", r"government", r"ministry",
    r"ministero", r"nederland", r"belgische staat", r"emprunt d.etat", r"irish government",
    r"european union",
    # regions and cities
    r"r[ée]gion\w*", r"reg", r"province", r"provincia", r"municipal\w*", r"city of", r"ville de",
    r"ville", r"county", r"kommune", r"fylkeskommune", r"landsting", r"communaut[ée]",
    r"d[ée]partement", r"m[ée]tropole", r"gemeenschap", r"assis\.? ?publ", r"ile de france",
    r"ile-de-france", r"syctom", r"sytral", r"tisseo", r"paris", r"lyon", r"marseille",
    # supranationals, agencies, public lenders
    r"eib", r"european investment bank", r"ibrd", r"world bank", r"ebrd", r"ifc",
    r"international finance corporation", r"kfw", r"asian development bank",
    r"asian infrastructure investment bank", r"inter-american development bank",
    r"new development bank", r"islamic development bank", r"idb", r"council of europe",
    r"nordic investment bank", r"fmo", r"rentenbank", r"kommunalbanken", r"bpifrance",
    r"agence", r"caisse amort", r"cades", r"caisse des d[ée]p[ôo]ts", r"caisse depots",
    r"caisse fse financement local", r"sfil", r"cassa depositi", r"unedic",
    r"action logement", r"soci[ée]t[ée] du grand paris", r"sncf", r"ratp",
    r"development bank", r"university", r"universit[ée]", r"hospital\w*", r"h[ôo]pitaux",
    # securitisations and funding vehicles. Generic words such as "funding" are
    # left out on purpose: listed groups like Land Securities use them too.
    r"clo", r"rmbs", r"securitisation", r"exmoor funding", r"pacific life global funding",
    r"pepper residential securities", r"finsbury square", r"green storm",
    r"pass through trust", r"sukuk", r"trust \d+",
]
_PUBLIC_RE = re.compile(r"\b(" + "|".join(_PUBLIC_PATTERNS) + r")\b")


def _normalise(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_bond_list(source) -> pd.DataFrame:
    """Read the Euronext ESG bond list (.xlsx, or a .csv with the same columns).

    The header row is found by looking for the ISIN column, so title rows
    above the table do not matter.
    """
    name = getattr(source, "name", str(source)).lower()
    raw = (pd.read_csv(source, header=None, dtype=str) if name.endswith(".csv")
           else pd.read_excel(source, header=None, dtype=str))

    header_row = next((i for i in range(min(len(raw), 40))
                       if any(_normalise(c) in ("isin", "isin code") for c in raw.iloc[i].dropna())), None)
    if header_row is None:
        raise ValueError("No header row with an 'ISIN' column was found.")

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [_normalise(c) for c in raw.iloc[header_row]]
    out = pd.DataFrame(index=df.index)
    for std, aliases in _COLUMN_ALIASES.items():
        hit = next((c for c in df.columns if c in aliases), None)
        out[std] = df[hit] if hit is not None else np.nan
    out = out.dropna(subset=["issuer", "isin"]).reset_index(drop=True)
    out["issuer"] = out["issuer"].str.strip()
    out["listing_date"] = _parse_dates(out["listing_date"])
    out["nominal_mm"] = out["nominal_mm"].map(_parse_number)
    out["type"] = out["type"].fillna("").str.strip()
    return out


def _parse_dates(col: pd.Series) -> pd.Series:
    # the Excel file has ISO dates, the website 27/03/2012
    iso = pd.to_datetime(col, errors="coerce", format="ISO8601")
    dmy = pd.to_datetime(col, errors="coerce", format="%d/%m/%Y")
    return iso.fillna(dmy)


def _parse_number(x) -> float:
    """1400, '1 000,00' (European style) or '1,000.00' (English style)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) != 3 else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def is_public_issuer(name: str) -> bool:
    return bool(_PUBLIC_RE.search(_normalise(name)))


def filter_bonds(bonds: pd.DataFrame, s: Settings) -> pd.DataFrame:
    b = bonds.copy()
    b["is_public"] = b["issuer"].map(is_public_issuer)
    wanted = {t.lower() for t in s.bond_types}
    b["type_short"] = (b["type"].str.lower().str.replace(" bonds", "")
                       .str.replace(" bond", "").str.strip())
    keep = b["type_short"].isin(wanted)
    keep &= b["listing_date"].between(pd.Timestamp(s.start), pd.Timestamp(s.end))
    keep &= b["nominal_mm"].fillna(0) >= s.min_nominal_mm
    if s.exclude_public_issuers:
        keep &= ~b["is_public"]
    return b[keep].reset_index(drop=True)


def first_issues(bonds: pd.DataFrame) -> pd.DataFrame:
    """One row per issuer: its first bond in the sample, plus count and total size."""
    g = bonds.sort_values("listing_date").groupby("issuer", as_index=False)
    return g.agg(event_date=("listing_date", "first"), first_isin=("isin", "first"),
                 n_bonds=("isin", "size"), total_nominal_mm=("nominal_mm", "sum"))


# =============================================================================
# 3  Finding share tickers
# =============================================================================

_LEGAL_FORMS = (r"\b(sa|se|nv|plc|ag|ab|asa|as|oyj|spa|ltd|limited|group|groupe|holding|holdings|"
                r"finance|financing|finco|treasury|international|entities|bv|gmbh|dac|sarl|sas|"
                r"corporation|corp|company|co|inc|llc|the|de|du|la|le|a/s|a-s|hf|pjsc|tas|aş)\b")

# Main European exchanges get a small bonus and secondary German venues or OTC
# quotes a penalty, so that GFC.PA wins over GI6A.F for Gecina.
_PRIMARY_EXCHANGES = {"PAR", "AMS", "BRU", "LIS", "MIL", "MCE", "STO", "OSL", "CPH", "HEL", "ICE",
                      "GER", "LSE", "VIE", "EBS", "IST", "ATH", "ISE", "WSE", "PRA", "BUD", "NYQ", "NMS"}
_SECONDARY_EXCHANGES = {"FRA", "DUS", "MUN", "STU", "BER", "HAM", "HAN", "PNK", "OQX", "OQB"}
_BOND_LIKE = re.compile(r"%|\d+/\d+|^[A-Z]{2}[A-Z0-9]{9}\d")   # coupons, maturities, ISIN-like symbols


def _strip_accents(s: str) -> str:
    import unicodedata
    s = s.replace("ø", "o").replace("Ø", "O").replace("æ", "ae").replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def clean_company_name(name: str) -> str:
    n = _strip_accents(_normalise(name))
    n = n.replace("société anonyme", " ").replace("societe anonyme", " ")
    n = re.sub(r"\(.*?\)", " ", n)
    n = n.replace(".", "")
    n = re.sub(r"[^\w\s&'/-]", " ", n)
    n = re.sub(_LEGAL_FORMS, " ", n)
    return re.sub(r"\s+", " ", n).strip()


def name_similarity(a: str, b: str) -> float:
    a, b = clean_company_name(a), clean_company_name(b)
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    contained = len(small & big) / len(small)
    # "gecina" vs "gecina sa": every word of the short name is in the long one
    return max(ratio, 0.95 * contained if contained == 1.0 else 0.8 * contained * ratio + 0.2 * ratio)


def score_quote(issuer: str, q: dict) -> float:
    """How likely a Yahoo search result is the share of this issuer (0 to 1)."""
    if q.get("quoteType") != "EQUITY":
        return 0.0
    qname = q.get("longname") or q.get("shortname") or ""
    if _BOND_LIKE.search(qname) or _BOND_LIKE.search(q.get("symbol", "")):
        return 0.0
    a, b = clean_company_name(issuer), clean_company_name(qname)
    s = name_similarity(issuer, qname)
    if len(a) <= 5 and a not in b.split():     # short names must match a whole word
        s = min(s, 0.5)
    ex = q.get("exchange", "")
    if ex in _PRIMARY_EXCHANGES:
        s += 0.05
    elif ex in _SECONDARY_EXCHANGES:
        s -= 0.15
    return round(min(s, 1.0), 3)


def _yahoo_search(query: str) -> list[dict]:
    import yfinance as yf
    try:
        return yf.Search(query, max_results=8, news_count=0, raise_errors=False).quotes or []
    except Exception:
        return []


def suggest_tickers(issuers: list[str], search=_yahoo_search, cache_path: str | None = None,
                    progress=None) -> pd.DataFrame:
    """Search Yahoo for each issuer and propose a ticker with a 0-1 score.

    Rows with a low score must be checked by hand before they are used.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    rows = []
    for i, issuer in enumerate(issuers):
        if progress:
            progress(i, len(issuers), issuer)
        quotes = cache.get(issuer)
        if quotes is None:
            quotes = cache[issuer] = search(clean_company_name(issuer) or issuer)
        best, best_score = None, 0.0
        for q in quotes:
            sc = score_quote(issuer, q)
            if sc > best_score:
                best, best_score = q, sc
        rows.append({"issuer": issuer,
                     "ticker": best.get("symbol") if best else "",
                     "quote_name": (best.get("longname") or best.get("shortname")) if best else "",
                     "exchange": best.get("exchange", "") if best else "",
                     "score": round(best_score, 2),
                     "include": bool(best) and best_score >= 0.75})
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    return pd.DataFrame(rows)


# =============================================================================
# 4  Market data: prices, factors, exchange rates, firm info, earnings dates
# =============================================================================

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRENCH_FILES = {
    ("Europe", "monthly"): "Europe_5_Factors",
    ("Europe", "daily"): "Europe_5_Factors_Daily",
    ("Developed", "monthly"): "Developed_5_Factors",
    ("Developed", "daily"): "Developed_5_Factors_Daily",
}
ECB_FX_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"

# Trading currency by Yahoo ticker suffix. Used when firm_info.csv has no
# currency for a ticker. London prices are quoted in pence (GBp); the factor
# of 100 does not change returns, so pence are treated as pounds.
SUFFIX_CURRENCY = {
    "PA": "EUR", "AS": "EUR", "BR": "EUR", "LS": "EUR", "MI": "EUR", "MC": "EUR", "DE": "EUR",
    "F": "EUR", "DU": "EUR", "MU": "EUR", "HA": "EUR", "BE": "EUR", "SG": "EUR", "HE": "EUR",
    "IR": "EUR", "VI": "EUR", "AT": "EUR",
    "OL": "NOK", "ST": "SEK", "CO": "DKK", "IC": "ISK", "L": "GBP", "SW": "CHF", "WA": "PLN",
    "RO": "RON", "IS": "TRY", "T": "JPY", "TO": "CAD", "V": "CAD", "NE": "CAD", "MX": "MXN",
    "AX": "AUD", "KS": "KRW", "KQ": "KRW", "AE": "AED", "QA": "QAR", "SR": "SAR", "": "USD",
}
# Currencies with a fixed rate to the dollar, which the ECB does not publish.
USD_PEGS = {"AED": 3.6725, "QAR": 3.64, "SAR": 3.75}

# Country names (as Yahoo writes them) -> region used in matching.
REGIONS = {
    "Nordics": ["Norway", "Sweden", "Denmark", "Finland", "Iceland"],
    "Western Europe": ["France", "Germany", "Netherlands", "Belgium", "Luxembourg", "Austria",
                       "Switzerland", "Liechtenstein", "Monaco"],
    "UK and Ireland": ["United Kingdom", "Ireland", "Guernsey", "Jersey", "Isle of Man"],
    "Southern Europe": ["Italy", "Spain", "Portugal", "Greece", "Cyprus", "Malta"],
    "Central and Eastern Europe": ["Poland", "Romania", "Czech Republic", "Hungary", "Turkey",
                                   "Slovenia", "Croatia", "Bulgaria"],
}
COUNTRY_REGION = {c: r for r, cs in REGIONS.items() for c in cs}


def ticker_suffix(ticker: str) -> str:
    return ticker.rsplit(".", 1)[1] if "." in ticker else ""


def download_prices(tickers, start: str, end: str | None = None) -> pd.DataFrame:
    """Daily adjusted close prices (dividends included) from Yahoo Finance."""
    import yfinance as yf
    tickers = sorted({t for t in tickers if isinstance(t, str) and t})
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if data is None or data.empty:
        raise ConnectionError("Yahoo Finance returned no data. Check the internet connection "
                              "and try again.")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if not isinstance(data.columns, pd.MultiIndex):
        close.columns = tickers
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all", axis=1)


def parse_french_csv(text: str) -> pd.DataFrame:
    """Read a Kenneth French factor CSV. Keeps the first table (monthly YYYYMM
    or daily YYYYMMDD rows), skips the annual table below it, and turns
    percent into decimals."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "Mkt-RF" in l)
    header = [h.strip() for h in lines[start].split(",")]
    header[0] = "date"
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit():
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=header[:len(rows[0])])
    fmt = "%Y%m" if len(df["date"].iloc[0]) == 6 else "%Y%m%d"
    idx = pd.to_datetime(df["date"], format=fmt)
    if fmt == "%Y%m":
        idx = idx + pd.offsets.MonthEnd(0)
    df = df.drop(columns="date").astype(float).replace(-99.99, np.nan) / 100.0
    df.index = idx
    df.index.name = "date"
    return df


def download_french_factors(region: str = "Europe", freq: str = "monthly") -> pd.DataFrame:
    import requests
    r = requests.get(FRENCH_URL.format(name=FRENCH_FILES[(region, freq)]), timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return parse_french_csv(z.read(z.namelist()[0]).decode("latin-1"))


def read_french_file(path) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        return parse_french_csv(z.read(z.namelist()[0]).decode("latin-1"))
    with open(path, encoding="latin-1") as f:
        return parse_french_csv(f.read())


def read_prices_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def parse_ecb_fx(text: str) -> pd.DataFrame:
    """ECB reference rates: units of each currency per 1 euro, one row per day."""
    df = pd.read_csv(io.StringIO(text), na_values=["N/A"])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index().astype(float)


def download_ecb_fx() -> pd.DataFrame:
    import requests
    r = requests.get(ECB_FX_URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return parse_ecb_fx(z.read(z.namelist()[0]).decode("utf-8"))


def read_fx_file(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index().astype(float)


def read_firm_info(path) -> pd.DataFrame:
    """ticker, name, country, sector, industry, currency, source."""
    return pd.read_csv(path, dtype=str).fillna("")


def download_firm_info(tickers, progress=None) -> pd.DataFrame:
    """Country, sector and trading currency of each ticker from Yahoo (slow: one call per ticker)."""
    import yfinance as yf
    rows = []
    for i, t in enumerate(tickers):
        if progress:
            progress(i, len(tickers), t)
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        rows.append({"ticker": t, "name": info.get("longName", ""), "country": info.get("country", ""),
                     "sector": info.get("sector", ""), "industry": info.get("industry", ""),
                     "currency": info.get("currency", ""), "source": "yahoo" if info else ""})
    return pd.DataFrame(rows)


def download_earnings_dates(tickers, limit: int = 100, progress=None) -> pd.DataFrame:
    """Past earnings release dates from Yahoo, one row per (ticker, date).

    Yahoo's history is long for large firms and short for some small ones, so
    the event study only uses it for events after a firm's first known date.
    """
    import yfinance as yf
    rows = []
    for i, t in enumerate(tickers):
        if progress:
            progress(i, len(tickers), t)
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=limit)
        except Exception:
            ed = None
        if ed is None or ed.empty:
            continue
        for d in pd.to_datetime(ed.index).tz_localize(None).normalize().unique():
            rows.append({"ticker": t, "date": d})
    return pd.DataFrame(rows, columns=["ticker", "date"])


def read_earnings_file(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna()


def read_announcement_file(path) -> pd.DataFrame:
    """ticker, announcement_date (rows with an empty date are ignored)."""
    df = pd.read_csv(path, dtype=str)
    df["announcement_date"] = pd.to_datetime(df.get("announcement_date"), errors="coerce")
    return df.dropna(subset=["ticker", "announcement_date"])


# =============================================================================
# 5  Returns, currency conversion and control firms
# =============================================================================

def ticker_currencies(tickers, info: pd.DataFrame | None = None) -> pd.Series:
    """Trading currency of each ticker: firm_info.csv first, then the ticker suffix."""
    known = {}
    if info is not None and not info.empty:
        known = dict(zip(info["ticker"], info["currency"].str.upper()))   # "GBp" -> "GBP"
    out = {}
    for t in tickers:
        c = known.get(t) or SUFFIX_CURRENCY.get(ticker_suffix(t), "")
        out[t] = c
    return pd.Series(out, dtype=object)


def usd_per_unit(fx_per_eur: pd.DataFrame, currency: str) -> pd.Series | None:
    """Daily US dollars per one unit of ``currency``, from ECB euro rates."""
    usd = fx_per_eur["USD"]
    if currency == "USD":
        return pd.Series(1.0, index=fx_per_eur.index)
    if currency == "EUR":
        return usd
    if currency in USD_PEGS:
        return pd.Series(1.0 / USD_PEGS[currency], index=fx_per_eur.index)
    if currency in fx_per_eur:
        return usd / fx_per_eur[currency]
    return None


def convert_to_usd(prices: pd.DataFrame, fx_per_eur: pd.DataFrame,
                   info: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Multiply each price series by the dollar value of its currency on that day.

    Exchange rates are carried forward over at most 5 days (ECB holidays).
    Returns the converted prices and the tickers that could not be converted
    (unknown currency), which are dropped.
    """
    cur = ticker_currencies(prices.columns, info)
    fx = fx_per_eur.reindex(fx_per_eur.index.union(prices.index)).ffill(limit=5).reindex(prices.index)
    out, dropped = {}, []
    for c, tickers in cur.groupby(cur).groups.items():
        rate = usd_per_unit(fx, c)
        if rate is None:
            dropped += list(tickers)
            continue
        for t in tickers:
            out[t] = prices[t] * rate
    dropped += list(cur[cur == ""].index)
    return pd.DataFrame(out, index=prices.index)[[t for t in prices.columns if t in out]], dropped


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    m = prices.resample("ME").last()
    return m.pct_change(fill_method=None).iloc[1:]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns, each share measured from its own previous trading day.

    Weekend rows are dropped first: Gulf markets trade on Sundays, and a
    Sunday row would otherwise wipe out the Monday return of every European
    share (version 1 lost those Mondays). A holiday in one market gives a
    missing return that day and a two-day return the next.
    """
    p = prices[prices.index.dayofweek < 5]
    r = p.ffill(limit=5).pct_change(fill_method=None)
    return r.where(p.notna()).iloc[1:]


def winsorise(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Clip each month's cross-section at the pct and 100-pct percentiles."""
    if pct <= 0:
        return df
    lo = df.quantile(pct / 100, axis=1)
    hi = df.quantile(1 - pct / 100, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def beta_and_vol(window: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """Market beta, monthly volatility and number of months for every column.

    Each column uses only the months where it and the market both have data.
    """
    R = window.to_numpy(dtype=float)
    m = mkt.reindex(window.index).to_numpy(dtype=float)[:, None]
    ok = ~np.isnan(R) & ~np.isnan(m)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_bar = np.where(ok, R, 0.0).sum(0) / n
        m_bar = np.where(ok, m, 0.0).sum(0) / n
        dr = np.where(ok, R - r_bar, 0.0)
        dm = np.where(ok, m - m_bar, 0.0)
        beta = (dr * dm).sum(0) / (dm ** 2).sum(0)
        vol = np.sqrt((dr ** 2).sum(0) / (n - 1))
    return pd.DataFrame({"beta": beta, "vol": vol, "n": n}, index=window.columns)


def firm_countries(tickers, info: pd.DataFrame | None) -> pd.Series:
    known = dict(zip(info["ticker"], info["country"])) if info is not None and not info.empty else {}
    return pd.Series({t: known.get(t, "") for t in tickers}, dtype=object)


def firm_sectors(tickers, info: pd.DataFrame | None) -> pd.Series:
    known = (dict(zip(info["ticker"], info["sector"])) if info is not None and not info.empty
             and "sector" in info else {})
    return pd.Series({t: known.get(t, "") for t in tickers}, dtype=object)


def match_controls(events: pd.DataFrame, rets_m: pd.DataFrame, mkt_m: pd.Series,
                   universe: list[str], s: Settings, info: pd.DataFrame | None = None) -> pd.DataFrame:
    """Give every issuer one control firm (its "twin") that never issued a green bond.

    Similarity is measured on market beta and monthly volatility over the
    months before the issuer's first green bond, as a standardised distance.

    With match_mode "country_region" the twin is the nearest firm from the same
    country, if its distance is within ``match_caliper``; otherwise the nearest
    from the same region within the caliper; otherwise the nearest firm in the
    whole pool. "sector_country" puts the same sector (Yahoo's 11 sectors) in
    front: same sector and country, same sector and region, same sector
    anywhere, then the country/region/pool steps. With "pool", it is always the
    nearest firm in the whole pool.
    Issuers are matched in date order and a twin is used once.
    """
    issuers = set(events["ticker"])
    candidates = [t for t in universe if t in rets_m.columns and t not in issuers]
    country = firm_countries(set(candidates) | issuers, info)
    region = country.map(lambda c: COUNTRY_REGION.get(c, "Rest of world" if c else ""))
    sector = firm_sectors(set(candidates) | issuers, info)
    used, rows = set(), []

    for _, e in events.sort_values("event_date").iterrows():
        t = e["ticker"]
        t0 = pd.Timestamp(e["event_date"]) + pd.offsets.MonthEnd(0)
        window = rets_m.loc[(rets_m.index < t0) &
                            (rets_m.index >= t0 - pd.DateOffset(months=s.match_window_months))]
        row = {"ticker": t, "control": None, "match_level": None, "distance": np.nan,
               "country_issuer": country.get(t, ""), "country_control": "",
               "sector_issuer": sector.get(t, ""), "sector_control": ""}
        if t not in window or window[t].count() < s.match_min_months:
            rows.append(row)
            continue
        own = beta_and_vol(window[[t]], mkt_m).iloc[0]
        pool = [c for c in candidates if s.match_with_replacement or c not in used]
        st = beta_and_vol(window[pool], mkt_m) if pool else pd.DataFrame(columns=["beta", "vol", "n"])
        st = st[st["n"] >= s.match_min_months].dropna()
        if st.empty:
            rows.append(row)
            continue
        # distances are scaled by the spread of the whole pool, so they mean the
        # same thing in every tier
        sb, sv = st["beta"].std() or 1.0, st["vol"].std() or 1.0
        st["d"] = np.sqrt(((st["beta"] - own["beta"]) / sb) ** 2 + ((st["vol"] - own["vol"]) / sv) ** 2)

        c, r, sec = country.get(t, ""), region.get(t, ""), sector.get(t, "")
        same_c = country.reindex(st.index).eq(c) if c else pd.Series(False, index=st.index)
        same_r = (region.reindex(st.index).eq(r) if r and r != "Rest of world"
                  else pd.Series(False, index=st.index))
        same_s = sector.reindex(st.index).eq(sec) if sec else pd.Series(False, index=st.index)
        tiers = [("pool", st)]
        if s.match_mode == "country_region":
            tiers = [("country", st[same_c]), ("region", st[same_r]), ("pool", st)]
        elif s.match_mode == "sector_country":
            tiers = [("sector+country", st[same_s & same_c]), ("sector+region", st[same_s & same_r]),
                     ("sector", st[same_s]), ("country", st[same_c]), ("region", st[same_r]), ("pool", st)]
        for level, cand in tiers:
            if cand.empty:
                continue
            best = cand["d"].idxmin()
            if level == "pool" or cand.loc[best, "d"] <= s.match_caliper:
                used.add(best)
                row.update(control=best, match_level=level, distance=float(cand.loc[best, "d"]),
                           country_control=country.get(best, ""), sector_control=sector.get(best, ""),
                           beta_issuer=own["beta"], beta_control=st.loc[best, "beta"],
                           vol_issuer=own["vol"], vol_control=st.loc[best, "vol"])
                break
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 6  Calendar-time portfolios and factor regressions
# =============================================================================

def holding_mask(events: pd.DataFrame, index: pd.DatetimeIndex, s: Settings,
                 key: str = "ticker") -> pd.DataFrame:
    """1 in the months a firm is held, 0 otherwise (months x firms)."""
    cols = list(events[key].dropna().unique())
    mask = pd.DataFrame(0.0, index=index, columns=cols)
    for _, e in events.dropna(subset=[key]).iterrows():
        listing_month = pd.Timestamp(e["event_date"]) + pd.offsets.MonthEnd(0)
        first = listing_month + pd.offsets.MonthEnd(s.entry_lag_months)
        last = first + pd.offsets.MonthEnd(s.hold_months - 1) if s.hold_months > 0 else index.max()
        mask.loc[(index >= first) & (index <= last), e[key]] = 1.0
    return mask


def portfolio_returns(rets_m: pd.DataFrame, mask: pd.DataFrame, min_firms: int) -> pd.DataFrame:
    """Equal-weighted monthly return and number of firms held."""
    cols = [c for c in mask.columns if c in rets_m.columns]
    r = rets_m.reindex(mask.index)[cols]
    held = mask[cols].where(r.notna(), 0.0)
    n = held.sum(axis=1)
    ret = (r.fillna(0.0) * held).sum(axis=1) / n.replace(0, np.nan)
    out = pd.DataFrame({"ret": ret, "n_firms": n})
    return out[out["n_firms"] >= min_firms]


def factor_regression(excess: pd.Series, factors: pd.DataFrame, model: str, nw_lags: int = 6) -> dict:
    """OLS of a monthly excess return on a factor model, Newey-West errors.

    Alpha is given per month and per year (x12), in percent.
    """
    cols = FACTOR_MODELS[model]
    data = pd.concat([excess.rename("y"), factors[cols]], axis=1, join="inner").dropna()
    if len(data) < len(cols) + 12:
        return {"model": model, "n_months": len(data), "error": "too few months"}
    res = sm.OLS(data["y"], sm.add_constant(data[cols])).fit(cov_type="HAC",
                                                             cov_kwds={"maxlags": nw_lags})
    out = {"model": model, "n_months": int(len(data)),
           "alpha_m_pct": 100 * res.params["const"], "alpha_ann_pct": 1200 * res.params["const"],
           "t_alpha": res.tvalues["const"], "p_alpha": res.pvalues["const"], "r2": res.rsquared}
    for c in cols:
        out[f"b_{c}"] = res.params[c]
        out[f"t_{c}"] = res.tvalues[c]
    return out


def performance_summary(ret: pd.Series, rf: pd.Series | None = None) -> dict:
    r = ret.dropna()
    if len(r) < 2:
        return {"months": len(r)}
    ex = r - rf.reindex(r.index).fillna(0) if rf is not None else r
    vol = r.std() * np.sqrt(12)
    return {"months": len(r), "mean_m_pct": 100 * r.mean(),
            "ann_return_pct": 100 * ((1 + r).prod() ** (12 / len(r)) - 1),
            "ann_vol_pct": 100 * vol,
            "sharpe": ex.mean() * 12 / vol if vol > 0 else np.nan,
            "max_drawdown_pct": 100 * ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()}


# =============================================================================
# 7  Event study
# =============================================================================

def apply_event_dates(events: pd.DataFrame, announcements: pd.DataFrame | None,
                      s: Settings) -> pd.DataFrame:
    """Use the bond's announcement date instead of its listing date where one is known.

    The listing date is kept in ``listing_date`` and ``date_source`` says which
    date each event uses.
    """
    ev = events.copy()
    ev["listing_date"] = ev["event_date"]
    ev["date_source"] = "listing"
    if s.event_date_source != "announcement" or announcements is None or announcements.empty:
        return ev
    ann = announcements.drop_duplicates("ticker").set_index("ticker")["announcement_date"]
    a = ev["ticker"].map(ann)
    use = a.notna() & (a <= ev["listing_date"])        # an announcement after listing is a typo
    ev.loc[use, "event_date"] = a[use]
    ev.loc[use, "date_source"] = "announcement"
    return ev


def earnings_flags(events: pd.DataFrame, earnings: pd.DataFrame | None, dates: pd.DatetimeIndex,
                   s: Settings) -> pd.Series:
    """Per event: "clash" (earnings release inside the window), "clear", or "unknown".

    "unknown" means Yahoo has no earnings history for the firm that far back,
    so we cannot tell.
    """
    flags = pd.Series("unknown", index=events.index, dtype=object)
    if earnings is None or earnings.empty:
        return flags
    by_ticker = {t: pd.DatetimeIndex(g["date"]).sort_values() for t, g in earnings.groupby("ticker")}
    lo, hi = s.earnings_window
    for i, e in events.iterrows():
        ed = by_ticker.get(e["ticker"])
        if ed is None or len(ed) == 0:
            continue
        pos = dates.searchsorted(pd.Timestamp(e["event_date"]))
        if pos + lo < 0 or pos + hi >= len(dates):
            continue
        first_day, last_day = dates[pos + lo], dates[pos + hi]
        if ed.min() > first_day - pd.Timedelta(days=120):
            continue                                    # history starts too late to judge
        clash = ((ed >= first_day) & (ed <= last_day)).any()
        flags[i] = "clash" if clash else "clear"
    return flags


def _car_summary(per_event: pd.DataFrame, method: str, windows) -> list[dict]:
    out = []
    for w in windows:
        col = f"CAR[{w[0]},{w[1]}]"
        x = per_event.loc[per_event["method"] == method, col].dropna()
        if len(x) < 3:
            continue
        out.append({"method": method, "window": col, "n_events": len(x),
                    "mean_CAR_pct": 100 * x.mean(), "median_CAR_pct": 100 * x.median(),
                    "t_stat": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
                    "share_positive": (x > 0).mean()})
    return out


def event_study(events: pd.DataFrame, rets_d: pd.DataFrame, mkt_d: pd.Series, s: Settings,
                earnings: pd.DataFrame | None = None, path_window: tuple = (-10, 10)) -> dict:
    """Abnormal returns around each issuer's event date, two ways.

    market: actual return minus a market model (alpha + beta x market),
            estimated on days est_window before the event.
    twin:   the issuer's return minus its matched twin's return on the same
            day. Anything that hits both firms (market, sector, country,
            currency) cancels.

    CARs are summed over each event window and tested with a cross-sectional
    t-test. Events with an earnings release inside earnings_window are dropped
    when s.exclude_earnings is on and earnings dates are available.
    """
    dates = rets_d.index
    lo_all = min(s.est_window[0], path_window[0], *(w[0] for w in s.event_windows))
    hi_all = max(path_window[1], *(w[1] for w in s.event_windows))
    rel = np.arange(lo_all, hi_all + 1)
    mkt = mkt_d.reindex(dates).to_numpy()
    flags = earnings_flags(events, earnings, dates, s)

    rows, paths = [], {"market": [], "twin": []}
    for i, e in events.dropna(subset=["ticker"]).iterrows():
        t = e["ticker"]
        if t not in rets_d:
            continue
        pos = dates.searchsorted(pd.Timestamp(e["event_date"]))   # first trading day on/after the date
        if pos + lo_all < 0 or pos + hi_all >= len(dates):
            continue
        base = {"issuer": e.get("issuer", ""), "ticker": t, "event_date": e["event_date"],
                "date_source": e.get("date_source", "listing"), "earnings": flags.get(i, "unknown")}
        if s.exclude_earnings and base["earnings"] == "clash":
            continue
        r = rets_d[t].to_numpy()
        est = slice(pos + s.est_window[0], pos + s.est_window[1] + 1)
        ok = ~np.isnan(r[est]) & ~np.isnan(mkt[est])
        if ok.sum() < s.event_min_est_days:
            continue

        b, a = np.polyfit(mkt[est][ok], r[est][ok], 1)
        ar = pd.Series(r[pos + rel] - (a + b * mkt[pos + rel]), index=rel)
        rows.append({**base, "method": "market", "control": "", "beta": b,
                     **{f"CAR[{w[0]},{w[1]}]": ar.loc[w[0]:w[1]].sum(min_count=1) for w in s.event_windows}})
        paths["market"].append(ar.loc[path_window[0]:path_window[1]])

        c = e.get("control")
        if isinstance(c, str) and c in rets_d:
            ar_t = pd.Series(r[pos + rel] - rets_d[c].to_numpy()[pos + rel], index=rel)
            if ar_t.loc[path_window[0]:path_window[1]].notna().mean() > 0.8:
                rows.append({**base, "method": "twin", "control": c, "beta": np.nan,
                             **{f"CAR[{w[0]},{w[1]}]": ar_t.loc[w[0]:w[1]].sum(min_count=1)
                                for w in s.event_windows}})
                paths["twin"].append(ar_t.loc[path_window[0]:path_window[1]])

    per_event = pd.DataFrame(rows)
    summary = []
    if not per_event.empty:
        for method in ("market", "twin"):
            summary += _car_summary(per_event, method, s.event_windows)
    caar = pd.DataFrame({m: pd.concat(p, axis=1).mean(axis=1).cumsum() for m, p in paths.items() if p})
    counts = flags.value_counts().to_dict()
    return {"per_event": per_event, "summary": pd.DataFrame(summary), "caar_path": caar,
            "earnings_check": {k: int(counts.get(k, 0)) for k in ("clash", "clear", "unknown")}}


# =============================================================================
# 8  Full run, comparison of designs, robustness grid
# =============================================================================

def prepare_returns(prices: pd.DataFrame, s: Settings, fx: pd.DataFrame | None,
                    info: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Prices in the currency the settings ask for, plus tickers dropped on the way."""
    if s.currency == "USD":
        if fx is None:
            raise ValueError("Converting to US dollars needs exchange rates (data/ecb_fx_rates.csv).")
        return convert_to_usd(prices, fx, info)
    return prices, []


def run_analysis(events: pd.DataFrame, prices: pd.DataFrame, factors_m: pd.DataFrame,
                 factors_d: pd.DataFrame | None, universe: list[str], s: Settings,
                 fx: pd.DataFrame | None = None, info: pd.DataFrame | None = None,
                 earnings: pd.DataFrame | None = None,
                 announcements: pd.DataFrame | None = None) -> dict:
    """Portfolios, regressions and the event study for one set of settings.

    ``events`` has one row per issuer with issuer, ticker and event_date
    (listing date of the first green bond).
    """
    prices, dropped = prepare_returns(prices, s, fx, info)
    events = events.dropna(subset=["ticker"])
    events = events[events["ticker"].isin(prices.columns)]
    # the same firm can appear under two issuer names (e.g. "BBVA" and "Banco
    # Bilbao Vizcaya Argentaria"): keep its earliest event
    events = events.sort_values("event_date").drop_duplicates("ticker").reset_index(drop=True)
    if len(events) < s.min_firms:
        raise ValueError(f"Only {len(events)} issuer(s) with share prices match these settings, fewer than "
                         f"the {s.min_firms} needed for a portfolio. Check the bond types, dates and size filter.")
    events = apply_event_dates(events, announcements, s)

    rets_m = winsorise(monthly_returns(prices), s.winsor_pct)
    fm = factors_m.reindex(rets_m.index)
    mkt_m = fm["Mkt-RF"] + fm["RF"]

    matches = match_controls(events, rets_m, mkt_m, universe, s, info)
    events = events.merge(matches[["ticker", "control", "match_level"]], on="ticker", how="left")

    idx = rets_m.index
    green = portfolio_returns(rets_m, holding_mask(events, idx, s, "ticker"), s.min_firms)
    matched = events.dropna(subset=["control"])
    control = (portfolio_returns(rets_m, holding_mask(matched, idx, s, "control"), s.min_firms)
               if not matched.empty else pd.DataFrame(columns=["ret", "n_firms"], dtype=float))
    rf = fm["RF"]
    ports = pd.DataFrame({"green": green["ret"], "control": control["ret"]}).dropna(how="all")
    ports = ports.loc[pd.Timestamp(s.start):pd.Timestamp(s.end) + pd.offsets.MonthEnd(s.hold_months + 1)]
    ports = ports.loc[:factors_m.index.max()]           # no months after the factor data ends
    ports["green_minus_control"] = ports["green"] - ports["control"]
    ports["n_green"] = green["n_firms"]
    ports["n_control"] = control["n_firms"]

    regs = []
    for name in ("green", "control", "green_minus_control"):
        y = ports[name].dropna()
        if y.empty:
            continue
        # the long-short return is already an excess return
        y_ex = y if name == "green_minus_control" else y - rf.reindex(y.index)
        for model in FACTOR_MODELS:
            regs.append({"portfolio": name, **factor_regression(y_ex, fm, model, s.nw_lags)})

    perf = pd.DataFrame([{"portfolio": n,
                          **performance_summary(ports[n], None if n == "green_minus_control" else rf)}
                         for n in ("green", "control", "green_minus_control")])

    es = None
    if factors_d is not None and not factors_d.empty:
        rets_d = daily_returns(prices)
        mkt_d = (factors_d["Mkt-RF"] + factors_d["RF"]).reindex(rets_d.index)
        es = event_study(events, rets_d, mkt_d, s, earnings)

    return {"events": events, "matches": matches, "portfolios": ports,
            "regressions": pd.DataFrame(regs), "performance": perf, "event_study": es,
            "settings": s, "dropped_tickers": dropped}


def headline(res: dict, model: str = "FF5") -> dict:
    """The few numbers worth comparing between runs."""
    r = res["regressions"]

    def pick(port, col):
        x = r[(r.portfolio == port) & (r.model == model)]
        return float(x[col].iloc[0]) if not x.empty and col in x else np.nan

    out = {"issuers_with_prices": int(len(res["events"])),
           "matched": int(res["matches"]["control"].notna().sum()),
           "months": int(pick("green", "n_months")) if pd.notna(pick("green", "n_months")) else 0,
           "issuer_alpha": pick("green", "alpha_ann_pct"), "issuer_t": pick("green", "t_alpha"),
           "issuer_beta_mkt": pick("green", "b_Mkt-RF"),
           "control_alpha": pick("control", "alpha_ann_pct"), "control_t": pick("control", "t_alpha"),
           "diff_alpha": pick("green_minus_control", "alpha_ann_pct"),
           "diff_t": pick("green_minus_control", "t_alpha")}
    lv = res["matches"]["match_level"].value_counts()
    out.update({f"twins_{k}": int(lv.get(k, 0)) for k in
                ("sector+country", "sector+region", "sector", "country", "region", "pool")})
    m = res["matches"].dropna(subset=["control"])
    out["twins_same_sector"] = int((m["sector_issuer"].ne("") & m["sector_issuer"].eq(m["sector_control"])).sum()) \
        if "sector_issuer" in m else 0
    es = res["event_study"]
    if es is not None and not es["summary"].empty:
        sm_ = es["summary"]
        for method in ("market", "twin"):
            x = sm_[(sm_.method == method) & (sm_.window == "CAR[-1,1]")]
            if not x.empty:
                out[f"car_m1p1_{method}"] = float(x["mean_CAR_pct"].iloc[0])
                out[f"car_m1p1_{method}_t"] = float(x["t_stat"].iloc[0])
    return out


DESIGNS = {
    "Version 1: local currency, twins from whole pool": dict(currency="local", match_mode="pool"),
    "Dollars, twins from whole pool": dict(currency="USD", match_mode="pool"),
    "V2-C: dollars, twins by country then region": dict(currency="USD", match_mode="country_region"),
    "V2-S (main): dollars, twins by sector first": dict(currency="USD", match_mode="sector_country"),
}


def compare_designs(events, prices, fm, fd, universe, s: Settings, **data) -> pd.DataFrame:
    """Run the designs side by side so the effect of each fix is visible."""
    rows = []
    for label, change in DESIGNS.items():
        res = run_analysis(events, prices, fm, fd, universe, replace(s, **change), **data)
        rows.append({"design": label, **headline(res, s.factor_model)})
    return pd.DataFrame(rows)


def robustness_grid(events, prices, factors_m, universe, s: Settings,
                    hold_list=(12, 24, 36, 60), winsor_list=(0.0, 1.0, 5.0), models=None,
                    **data) -> pd.DataFrame:
    """Issuers-minus-twins alpha for every holding period x winsorising level."""
    models = tuple(models or (s.factor_model,))
    data = {k: v for k, v in data.items() if k in ("fx", "info")}
    rows = []
    for h in hold_list:
        for w in winsor_list:
            r = run_analysis(events, prices, factors_m, None, universe,
                             replace(s, hold_months=h, winsor_pct=w), **data)["regressions"]
            for m in models:
                x = r[(r.portfolio == "green_minus_control") & (r.model == m)]
                if not x.empty and "alpha_ann_pct" in x and pd.notna(x["alpha_ann_pct"].iloc[0]):
                    rows.append({"model": m, "hold_months": h, "winsor_pct": w,
                                 "alpha_ann_pct": x["alpha_ann_pct"].iloc[0],
                                 "t_alpha": x["t_alpha"].iloc[0], "n_months": x["n_months"].iloc[0]})
    return pd.DataFrame(rows)


# =============================================================================
# 9  Demo data
# =============================================================================

def make_demo_data(seed: int = 7, n_issuers: int = 120, n_universe: int = 300,
                   green_alpha_ann: float = 0.02, announce_effect: float = 0.004,
                   idio_vol_daily: float = 0.012, start: str = "2010-01-01",
                   end: str = "2025-12-31") -> dict:
    """Made-up data in the same shape as the real files, with known true effects.

    Issuers earn ``green_alpha_ann`` extra per year after their first listing
    and ``announce_effect`` on the listing day. Prices are in US dollars and
    firms get random countries, so every part of the pipeline runs offline.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    T = len(days)
    fac = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0003, 0.011, T), "SMB": rng.normal(0.0, 0.005, T),
        "HML": rng.normal(0.0001, 0.005, T), "RMW": rng.normal(0.0001, 0.004, T),
        "CMA": rng.normal(0.0, 0.004, T), "RF": np.full(T, 0.00004),
    }, index=days)

    n = n_issuers + n_universe
    tickers = [f"GB{i:03d}" for i in range(n_issuers)] + [f"CT{i:03d}" for i in range(n_universe)]
    load = np.column_stack([rng.normal(1.0, 0.25, n), rng.normal(0.2, 0.4, n),
                            rng.normal(0.1, 0.4, n), rng.normal(0.0, 0.3, n), rng.normal(0.0, 0.3, n)])
    eps = rng.normal(0, 1, (T, n)) * rng.uniform(0.5, 1.5, n) * idio_vol_daily
    rets = fac["RF"].to_numpy()[:, None] + fac.iloc[:, :5].to_numpy() @ load.T + eps

    ev_days = rng.integers(int(T * 0.25), int(T * 0.85), n_issuers)
    for j, d in enumerate(ev_days):
        rets[d, j] += announce_effect
        rets[d:, j] += green_alpha_ann / 252
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=tickers)

    names = [f"Demo Industrial {i:03d} SA" for i in range(n_issuers)]
    bonds = pd.DataFrame({
        "Issuer": names + ["Republic of Demo", "Demo Region", "European Demo Bank"],
        "ISIN": [f"XS{rng.integers(10**9, 10**10)}" for _ in range(n_issuers + 3)],
        "Listing date": [days[d].strftime("%d/%m/%Y") for d in ev_days] + ["01/06/2018"] * 3,
        "Nominal (mm)": list(rng.choice([300, 500, 750, 1000], n_issuers)) + [3000, 500, 1000],
        "Currency": "EUR", "Type": "Green Bond", "Venue": "Euronext Paris",
    })
    mapping = pd.DataFrame({"issuer": names, "ticker": tickers[:n_issuers], "quote_name": names,
                            "exchange": "DEMO", "score": 1.0, "include": True})
    countries = ["France", "Italy", "Spain", "Norway", "Sweden", "Germany", "Netherlands"]
    info = pd.DataFrame({"ticker": tickers, "name": "", "country": rng.choice(countries, n),
                         "sector": "", "industry": "", "currency": "USD", "source": "demo"})
    fx = pd.DataFrame({"USD": 1.0}, index=days)          # euro = dollar in the demo world
    factors_m = (1 + fac).resample("ME").prod() - 1
    return {"bonds": bonds, "prices": prices, "factors_d": fac, "factors_m": factors_m,
            "mapping": mapping, "universe": tickers[n_issuers:], "info": info, "fx": fx,
            "true": {"green_alpha_ann_pct": 100 * green_alpha_ann,
                     "announce_effect_pct": 100 * announce_effect}}


def bonds_to_excel_bytes(bonds_raw: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([["ESG Bonds list (demo)"], [""]]).to_excel(xw, index=False, header=False)
        bonds_raw.to_excel(xw, index=False, startrow=2)
    return buf.getvalue()


def demo_events(demo: dict, s: Settings) -> pd.DataFrame:
    b = filter_bonds(load_bond_list(io.BytesIO(bonds_to_excel_bytes(demo["bonds"]))), s)
    return first_issues(b).merge(demo["mapping"][["issuer", "ticker"]], on="issuer", how="left")


# =============================================================================
# 10 Saved results
# =============================================================================
# Yahoo prices may not be republished, so the public copy of the project shows
# the real-data results from a JSON file instead of recomputing them.

def _frame_to_obj(df) -> dict:
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name if df.name is not None else "value")
    df = df.copy()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in date_cols:
        df[c] = df[c].dt.strftime("%Y-%m-%d")
    date_index = isinstance(df.index, pd.DatetimeIndex)
    index = df.index.strftime("%Y-%m-%d").tolist() if date_index else df.index.tolist()
    data = df.astype(object).where(df.notna(), None).values.tolist()
    return {"columns": [str(c) for c in df.columns], "index": index, "data": data,
            "date_columns": [str(c) for c in date_cols], "date_index": date_index}


def _obj_to_frame(o: dict) -> pd.DataFrame:
    df = pd.DataFrame(o["data"], columns=o["columns"])
    df.index = pd.to_datetime(o["index"]) if o["date_index"] else o["index"]
    for c in o["date_columns"]:
        df[c] = pd.to_datetime(df[c])
    df = df.infer_objects()
    for c in df.columns:
        if df[c].dtype == object and c not in o["date_columns"]:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df


def save_snapshot(res: dict, s: Settings, path, grid=None, issuers=None, comparison=None,
                  meta: dict | None = None) -> None:
    es = res["event_study"]
    frames = {"events": res["events"], "matches": res["matches"], "portfolios": res["portfolios"],
              "regressions": res["regressions"], "performance": res["performance"]}
    if es is not None:
        frames.update(es_per_event=es["per_event"], es_summary=es["summary"], es_caar_path=es["caar_path"])
    if grid is not None:
        frames["grid"] = grid
    if comparison is not None:
        frames["comparison"] = comparison
    if issuers is not None:
        frames["issuers"] = issuers[[c for c in ("issuer", "event_date", "n_bonds") if c in issuers]]
    meta = dict(meta or {})
    if es is not None:
        meta["earnings_check"] = es.get("earnings_check", {})
    out = {"format": "greenbond-snapshot-2", "settings": asdict(s), "meta": meta,
           "frames": {k: _frame_to_obj(v) for k, v in frames.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), default=str)


def load_snapshot(path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    fr = {k: _obj_to_frame(v) for k, v in raw["frames"].items()}
    s = settings_from_dict(raw["settings"])
    meta = raw.get("meta", {})
    es = None
    if "es_summary" in fr:
        caar = fr["es_caar_path"]
        caar.index = caar.index.astype(int)
        es = {"per_event": fr["es_per_event"], "summary": fr["es_summary"], "caar_path": caar,
              "earnings_check": meta.get("earnings_check", {})}
    res = {"events": fr["events"], "matches": fr["matches"], "portfolios": fr["portfolios"],
           "regressions": fr["regressions"], "performance": fr["performance"],
           "event_study": es, "settings": s, "dropped_tickers": meta.get("dropped_tickers", [])}
    return {"res": res, "settings": s, "grid": fr.get("grid"), "issuers": fr.get("issuers"),
            "comparison": fr.get("comparison"), "meta": meta}


def _pick(folder, *names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p):
            return p
    return None


def load_data_folder(folder: str, region: str = "Europe") -> dict:
    """Read every input file the analysis can use from one folder.

    Required: bond list, issuer_tickers.csv, control_universe.csv, prices,
    factor files. Optional: ecb_fx_rates.csv, firm_info.csv,
    earnings_dates.csv, announcement_dates.csv.
    """
    def need(*names):
        p = _pick(folder, *names)
        if p is None:
            raise FileNotFoundError(f"none of {names} in {folder}")
        return p

    mapping = pd.read_csv(need("issuer_tickers.csv"))
    mapping["include"] = mapping["include"].astype(str).str.lower().isin(["true", "1", "yes"])
    opt = lambda n, reader: reader(_pick(folder, n)) if _pick(folder, n) else None
    return {
        "bonds_all": load_bond_list(need("Euronext-Green-Bond-List.xlsx", "esg_bonds.xlsx", "esg_bonds.csv")),
        "mapping": mapping,
        "universe": pd.read_csv(need("control_universe.csv"))["ticker"].astype(str).tolist(),
        "prices": read_prices_csv(need("prices.csv.gz", "prices.csv")),
        "factors_m": read_french_file(need(f"{region}_5_Factors_CSV.zip", f"{region}_5_Factors.csv")),
        "factors_d": read_french_file(need(f"{region}_5_Factors_Daily_CSV.zip", f"{region}_5_Factors_Daily.csv")),
        "fx": opt("ecb_fx_rates.csv", read_fx_file),
        "info": opt("firm_info.csv", read_firm_info),
        "earnings": opt("earnings_dates.csv", read_earnings_file),
        "announcements": opt("announcement_dates.csv", read_announcement_file),
    }


def events_from(d: dict, s: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bonds = filter_bonds(d["bonds_all"], s)
    issuers = first_issues(bonds)
    events = issuers.merge(d["mapping"][d["mapping"]["include"]][["issuer", "ticker"]], on="issuer", how="left")
    return bonds, issuers, events


def build_snapshot(data_dir, out_path, s: Settings | None = None, region: str = "Europe") -> dict:
    """Run everything on the files in data_dir and save the results (no prices)."""
    s = s or Settings()
    d = load_data_folder(data_dir, region)
    bonds, issuers, events = events_from(d, s)
    extra = {k: d[k] for k in ("fx", "info", "earnings", "announcements")}
    res = run_analysis(events, d["prices"], d["factors_m"], d["factors_d"], d["universe"], s, **extra)
    grid = robustness_grid(events, d["prices"], d["factors_m"], d["universe"], s,
                           models=tuple(FACTOR_MODELS), **extra)
    comparison = compare_designs(events, d["prices"], d["factors_m"], d["factors_d"], d["universe"], s, **extra)
    public = filter_bonds(d["bonds_all"], replace(s, exclude_public_issuers=False))["is_public"].sum()
    in_pool = [t for t in d["universe"] if t in d["prices"].columns]
    fx = d["fx"]
    meta = {"computed_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "prices_to": d["prices"].dropna(how="all").index.max().strftime("%Y-%m-%d"),
            "factors_to": d["factors_m"].index.max().strftime("%Y-%m-%d"),
            "fx_to": fx.index.max().strftime("%Y-%m-%d") if fx is not None else "",
            "factor_region": region,
            "bonds_in_file": int(len(d["bonds_all"])), "bonds_after_filters": int(len(bonds)),
            "public_excluded": int(public) if s.exclude_public_issuers else 0,
            "issuers": int(len(issuers)), "control_universe": int(len(d["universe"])),
            "control_universe_with_prices": int(len(in_pool)),
            "announcement_dates_filled": int(len(d["announcements"])) if d["announcements"] is not None else 0,
            "earnings_dates_available": d["earnings"] is not None,
            "dropped_tickers": res["dropped_tickers"]}
    save_snapshot(res, s, out_path, grid=grid, issuers=issuers, comparison=comparison, meta=meta)
    return {"res": res, "comparison": comparison, "grid": grid, "meta": meta}


# =============================================================================
# 11 Command line
# =============================================================================

def update_data(data_dir: str, start: str = "2009-01-01") -> None:
    """Refresh the downloadable inputs in data_dir (needs internet; run on your own PC).

    1. ECB exchange rates          -> ecb_fx_rates.csv
    2. prices of tickers not yet in prices.csv.gz (e.g. the FTSE 100 and SMI firms)
    3. country and currency of tickers missing from firm_info.csv
    4. earnings dates of all issuers -> earnings_dates.csv
    5. Kenneth French factor files
    Every step is independent: if one fails, the others still run.
    """
    log = lambda m: print(m, flush=True)
    d = os.fspath(data_dir)
    s = Settings()

    try:
        fx_new = download_ecb_fx()
        path = os.path.join(d, "ecb_fx_rates.csv")
        if os.path.exists(path):         # keep filled gaps (e.g. ISK 2009-2017) from the old file
            fx_new = fx_new.combine_first(read_fx_file(path))
        fx_new.to_csv(path, float_format="%.6g")
        log(f"1. exchange rates: {len(fx_new)} days, to {fx_new.index.max():%Y-%m-%d}")
    except Exception as e:
        log(f"1. exchange rates: FAILED ({e})")

    mapping = pd.read_csv(os.path.join(d, "issuer_tickers.csv"))
    mapping = mapping[mapping["include"].astype(str).str.lower().isin(["true", "1", "yes"])]
    universe = pd.read_csv(os.path.join(d, "control_universe.csv"))["ticker"].astype(str).tolist()
    wanted = sorted(set(mapping["ticker"].dropna()) | set(universe))
    price_path = _pick(d, "prices.csv.gz", "prices.csv") or os.path.join(d, "prices.csv.gz")

    try:
        prices = read_prices_csv(price_path) if os.path.exists(price_path) else pd.DataFrame()
        missing = [t for t in wanted if t not in prices.columns]
        if missing:
            new = download_prices(missing, start)
            prices = pd.concat([prices, new], axis=1).sort_index()
            prices.to_csv(os.path.join(d, "prices.csv.gz"))
        log(f"2. prices: {len(missing)} new tickers requested, {prices.shape[1]} in file")
    except Exception as e:
        log(f"2. prices: FAILED ({e})")

    try:
        info_path = os.path.join(d, "firm_info.csv")
        info = read_firm_info(info_path) if os.path.exists(info_path) else pd.DataFrame(columns=["ticker"])
        todo = [t for t in wanted if t not in set(info.loc[info.get("source", "") == "yahoo", "ticker"])]
        if todo:
            new = download_firm_info(todo, progress=lambda i, n, t: print(f"   {i + 1}/{n} {t}", end="\r"))
            new = new[new["source"] == "yahoo"]
            if not new.empty:
                info = pd.concat([info[~info["ticker"].isin(new["ticker"])], new]).sort_values("ticker")
                info.to_csv(info_path, index=False)
            log(f"\n3. firm info: {len(new)} of {len(todo)} tickers found on Yahoo")
        else:
            log("3. firm info: nothing missing")
    except Exception as e:
        log(f"3. firm info: FAILED ({e})")

    try:
        issuers = sorted(mapping["ticker"].dropna().unique())
        ed = download_earnings_dates(issuers, progress=lambda i, n, t: print(f"   {i + 1}/{n} {t}", end="\r"))
        if ed.empty:
            log("\n4. earnings dates: FAILED (nothing came back from Yahoo)")
        else:
            ed.to_csv(os.path.join(d, "earnings_dates.csv"), index=False)
            log(f"\n4. earnings dates: {ed['ticker'].nunique()} of {len(issuers)} issuers have a history")
    except Exception as e:
        log(f"4. earnings dates: FAILED ({e})")

    for (region, freq), base in FRENCH_FILES.items():
        fname = f"{base}_CSV.zip"
        try:
            import requests
            r = requests.get(FRENCH_URL.format(name=base), timeout=60)
            r.raise_for_status()
            with open(os.path.join(d, fname), "wb") as f:
                f.write(r.content)
            log(f"5. French factors ({region}, {freq}): updated")
        except Exception as e:
            log(f"5. French factors ({region}, {freq}): FAILED ({e})")


def _print_results(res: dict, comparison: pd.DataFrame | None = None):
    pd.set_option("display.width", 180)
    r = res["regressions"]
    print("\nFactor regressions")
    print(r[["portfolio", "model", "n_months", "alpha_ann_pct", "t_alpha", "r2"]].round(3).to_string(index=False))
    if res["event_study"] is not None:
        print("\nEvent study")
        print(res["event_study"]["summary"].round(3).to_string(index=False))
    if comparison is not None:
        print("\nDesigns compared")
        print(comparison.round(2).T.to_string())


def selftest():
    """Checks of every building block:  python greenbond_engine.py --selftest"""
    results = []

    def check(name, ok):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name)

    check("European and English number formats",
          _parse_number("1 000,00") == 1000.0 and _parse_number("1,000.50") == 1000.5)
    check("public issuers flagged", all(map(is_public_issuer, ["EIB", "REPUBLIC OF ITALY", "VILLE DE PARIS",
                                                               "REGION WALLONNE", "KFW"])))
    check("listed groups kept", not any(map(is_public_issuer, ["LAND SECURITIES", "GECINA",
                                                               "ENEL FINANCE INTERNATIONAL"])))
    q = [{"symbol": "GI6A.F", "longname": "Gecina S.A.", "exchange": "FRA", "quoteType": "EQUITY"},
         {"symbol": "GFC.PA", "longname": "Gecina", "exchange": "PAR", "quoteType": "EQUITY"}]
    check("primary listing preferred", suggest_tickers(["GECINA"], search=lambda _: q)["ticker"][0] == "GFC.PA")
    f = parse_french_csv("x\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n199007,1.0,2.0,3.0,4.0,5.0,0.5\n\n Annual\n")
    check("French factor file parsed", f.index[0] == pd.Timestamp("1990-07-31") and f.iloc[0, 0] == 0.01)

    # currency: a Norwegian share flat in kroner while the krone gains 10% on
    # the dollar must show +10% in dollars
    days = pd.bdate_range("2020-01-01", periods=3)
    fx = pd.DataFrame({"USD": [1.10, 1.10, 1.10], "NOK": [10.0, 10.0, 10.0 / 1.1]}, index=days)
    px = pd.DataFrame({"EQNR.OL": [100.0, 100.0, 100.0], "ENGI.PA": [10.0, 11.0, 11.0],
                       "CBD.AE": [5.0, 5.0, 5.0]}, index=days)
    usd, dropped = convert_to_usd(px, fx)
    r = usd.pct_change().iloc[-1]
    check("currency conversion (krone +10%, euro share +10% in euros, dollar peg)",
          abs(r["EQNR.OL"] - 0.10) < 1e-9 and abs(usd["ENGI.PA"].pct_change().iloc[1] - 0.10) < 1e-9
          and abs(r["CBD.AE"]) < 1e-12 and not dropped)

    # matching: the same-country twin wins when it is within the caliper,
    # the closer foreign firm wins when the compatriot is too far away
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-31", periods=40, freq="ME")
    mkt = pd.Series(rng.normal(0.01, 0.04, 40), index=idx)
    base = rng.normal(0, 0.02, 40)
    rets = pd.DataFrame({"ISS": mkt + base, "FR_NEAR": mkt + base * 1.05, "IT_CLOSER": mkt + base * 1.0001,
                         "FR_FAR": 2.5 * mkt + base * 3}, index=idx)
    for k in range(30):
        rets[f"X{k}"] = rng.uniform(0.3, 1.8) * mkt + rng.normal(0, rng.uniform(0.01, 0.08), 40)
    info = pd.DataFrame({"ticker": list(rets.columns),
                         "country": ["France", "France", "Italy", "France"] + ["Germany"] * 30})
    ev = pd.DataFrame({"ticker": ["ISS"], "event_date": [pd.Timestamp("2018-06-15")]})
    m1 = match_controls(ev, rets, mkt, list(rets.columns[1:]), Settings(match_mode="country_region"), info).iloc[0]
    m2 = match_controls(ev, rets, mkt, ["IT_CLOSER", "FR_FAR"] + [f"X{k}" for k in range(30)],
                        Settings(match_mode="country_region"), info).iloc[0]
    m3 = match_controls(ev, rets, mkt, list(rets.columns[1:]), Settings(match_mode="pool"), info).iloc[0]
    info_s = info.assign(sector=["Utilities", "Banks", "Utilities", "Banks"] + ["Utilities"] * 30)
    m4 = match_controls(ev, rets, mkt, list(rets.columns[1:]), Settings(match_mode="sector_country"), info_s).iloc[0]
    check("matching: same sector first when asked", m4["sector_control"] == "Utilities"
          and m4["match_level"].startswith("sector"))
    check("matching: same country first, then wider", m1["control"] == "FR_NEAR" and m1["match_level"] == "country"
          and m2["match_level"] != "country" and m3["control"] == "IT_CLOSER")

    # whole pipeline recovers known effects on low-noise demo data
    alphas, car_m, car_t = [], [], []
    for seed in range(1, 5):
        d = make_demo_data(seed=seed, idio_vol_daily=0.002)
        st_ = Settings(start="2012-01-01", winsor_pct=0)
        res = run_analysis(demo_events(d, st_), d["prices"], d["factors_m"], d["factors_d"],
                           d["universe"], st_, fx=d["fx"], info=d["info"])
        r = res["regressions"]
        alphas.append(r[(r.portfolio == "green_minus_control") & (r.model == "FF5")]["alpha_ann_pct"].iloc[0])
        smy = res["event_study"]["summary"]
        car_m.append(smy[(smy.method == "market") & (smy.window == "CAR[-1,1]")]["mean_CAR_pct"].iloc[0])
        car_t.append(smy[(smy.method == "twin") & (smy.window == "CAR[-1,1]")]["mean_CAR_pct"].iloc[0])
    check(f"portfolio alpha recovered ({np.mean(alphas):.2f} vs true 2.00 % a year)", abs(np.mean(alphas) - 2) < 0.6)
    check(f"listing-day effect recovered, market model ({np.mean(car_m):.2f} vs 0.40 %)", abs(np.mean(car_m) - 0.4) < 0.15)
    check(f"listing-day effect recovered, twin-adjusted ({np.mean(car_t):.2f} vs 0.40 %)", abs(np.mean(car_t) - 0.4) < 0.2)

    # earnings filter: an event with an earnings release on the listing day is dropped
    ev0 = demo_events(d, st_).dropna(subset=["ticker"]).iloc[:5]
    ed = pd.DataFrame({"ticker": [ev0["ticker"].iloc[0]] * 2,
                       "date": [pd.Timestamp("2005-01-01"), d["prices"].index[d["prices"].index.searchsorted(ev0["event_date"].iloc[0])]]})
    out = event_study(ev0, daily_returns(d["prices"]), d["factors_d"]["Mkt-RF"] + d["factors_d"]["RF"],
                      st_, earnings=ed)
    check("earnings clash dropped", out["earnings_check"]["clash"] == 1
          and ev0["ticker"].iloc[0] not in set(out["per_event"]["ticker"]))

    # announcement dates replace listing dates when filled in
    ann = pd.DataFrame({"ticker": [ev0["ticker"].iloc[1]], "announcement_date": [ev0["event_date"].iloc[1] - pd.Timedelta(days=7)]})
    a = apply_event_dates(ev0, ann, Settings(event_date_source="announcement"))
    check("announcement date used", (a["date_source"] == "announcement").sum() == 1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "snap.json")
        save_snapshot(res, st_, fn)
        back = load_snapshot(fn)["res"]
    check("saved results reload unchanged",
          np.allclose(back["regressions"]["alpha_ann_pct"], res["regressions"]["alpha_ann_pct"], equal_nan=True)
          and back["portfolios"].index.equals(res["portfolios"].index)
          and np.allclose(back["event_study"]["caar_path"], res["event_study"]["caar_path"], equal_nan=True))
    print(f"\n{sum(results)} of {len(results)} checks passed.")
    if not all(results):
        raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Green bond issuers: share performance")
    ap.add_argument("--selftest", action="store_true", help="check the building blocks (offline)")
    ap.add_argument("--demo", action="store_true", help="run on made-up data with known effects")
    ap.add_argument("--update-data", action="store_true",
                    help="download exchange rates, missing prices, firm info, earnings dates and factors")
    ap.add_argument("--save-snapshot", metavar="OUT_JSON",
                    help="run on the files in --data-dir and save the results for the dashboard")
    ap.add_argument("--bonds", help="Euronext ESG bond list, for --suggest-tickers")
    ap.add_argument("--suggest-tickers", metavar="OUT_CSV", help="search Yahoo for each issuer's ticker")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    s = Settings()

    if args.selftest:
        selftest()
    elif args.update_data:
        update_data(args.data_dir)
    elif args.save_snapshot:
        out = build_snapshot(args.data_dir, args.save_snapshot, s)
        _print_results(out["res"], out["comparison"])
        print(f"\nSaved {args.save_snapshot}")
    elif args.demo:
        d = make_demo_data()
        s.start = "2012-01-01"
        res = run_analysis(demo_events(d, s), d["prices"], d["factors_m"], d["factors_d"],
                           d["universe"], s, fx=d["fx"], info=d["info"])
        print("Demo data. True values:", d["true"])
        _print_results(res)
    elif args.bonds and args.suggest_tickers:
        issuers = first_issues(filter_bonds(load_bond_list(args.bonds), s))["issuer"].tolist()
        print(f"{len(issuers)} issuers. Searching Yahoo Finance...")
        out = suggest_tickers(issuers, cache_path="ticker_search_cache.json",
                              progress=lambda i, n, x: print(f"  {i + 1}/{n} {x}"))
        out.to_csv(args.suggest_tickers, index=False)
        print(f"Saved {args.suggest_tickers}. Check rows with include = False or a low score by hand.")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
