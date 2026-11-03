"""
greenbond_engine.py
===================

Data and analysis engine for the question:

    Do investors benefit from holding the stocks of companies that issue
    green bonds?

The workflow mirrors the design of Badía, Cortez and Silva (2024, Economics
Letters): compare a portfolio of green bond issuers with a portfolio of matched
non-issuers and measure risk-adjusted performance with Fama-French factor
models. On top of that, an event study measures the share-price reaction on
the day a green bond is listed.

Data sources (all free):
    * Euronext "ESG Bonds list" (Excel download) -> bonds and issuers
    * Yahoo Finance via the yfinance package       -> daily stock prices
    * Kenneth French data library                  -> European factors

Everything is in this file except the dashboard (app.py). The file is
organised in numbered sections:

    1  Settings
    2  Loading and filtering the bond list
    3  Matching issuers to stock tickers
    4  Market data (prices and Fama-French factors)
    5  Returns, winsorising and matched control firms
    6  Calendar-time portfolios and factor regressions
    7  Event study
    8  Full pipeline and robustness grid
    9  Demo data (works offline)
    10 Command line

Author: Saeed Samadiye.
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass, field

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

BOND_TYPES = ("Green", "Sustainability", "Social", "Sustainability-Linked", "Blue", "Transition")


@dataclass
class Settings:
    """Every choice the analysis depends on. The dashboard edits these."""

    # sample
    start: str = "2014-01-01"
    end: str = "2025-12-31"
    bond_types: tuple = ("Green",)
    min_nominal_mm: float = 0.0          # minimum bond size, millions
    exclude_public_issuers: bool = True  # drop governments, agencies, supranationals

    # portfolios
    hold_months: int = 36                # months a firm stays in the portfolio after its first green bond
    entry_lag_months: int = 1            # start holding this many months after the listing month
    min_firms: int = 5                   # months with fewer firms in the portfolio are dropped
    winsor_pct: float = 1.0              # winsorise monthly stock returns at pct / 100-pct (0 = off)

    # matching
    match_window_months: int = 36        # pre-event window used to measure beta and volatility
    match_min_months: int = 18
    match_with_replacement: bool = False

    # regressions
    factor_model: str = "FF5"
    nw_lags: int = 6

    # event study (trading days relative to the listing date)
    est_window: tuple = (-250, -30)
    event_windows: tuple = ((-1, 1), (0, 5), (-5, 5))
    event_min_est_days: int = 120

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# =============================================================================
# 2  Loading and filtering the bond list
# =============================================================================

# Column names we need, and the spellings we accept for each.
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

# Patterns that mark an issuer as public, supranational or a securitisation
# vehicle rather than a stock-listed company. Matched as whole words on the
# lower-case name. Extend the list if you find issuers it misses.
_NON_LISTED_PATTERNS = [
    # states and public bodies
    r"republic", r"repubblica", r"kingdom", r"regno", r"state of", r"etat", r"government", r"ministry",
    r"ministero", r"nederland", r"belgische staat", r"emprunt d.etat",
    r"irish government", r"european union",
    # regions and cities
    r"r[ée]gion\w*", r"reg", r"province", r"provincia", r"municipal\w*", r"city of", r"ville de",
    r"ville", r"county", r"kommune", r"fylkeskommune", r"landsting", r"communaut[ée]",
    r"d[ée]partement", r"m[ée]tropole", r"gemeenschap", r"assis\.? ?publ", r"ile de france",
    r"ile-de-france", r"syctom", r"sytral", r"tisseo", r"paris", r"lyon", r"marseille",
    # supranational and public development / agency lenders
    r"eib", r"european investment bank", r"ibrd", r"world bank", r"ebrd", r"ifc",
    r"international finance corporation", r"kfw", r"asian development bank",
    r"asian infrastructure investment bank", r"inter-american development bank",
    r"new development bank", r"islamic development bank", r"idb", r"council of europe",
    r"nordic investment bank", r"fmo", r"rentenbank", r"kommunalbanken", r"bpifrance",
    r"agence", r"caisse amort", r"cades", r"caisse des d[ée]p[ôo]ts", r"caisse depots",
    r"caisse fse financement local", r"sfil", r"cassa depositi", r"unedic",
    r"action logement", r"soci[ée]t[ée] du grand paris", r"sncf", r"ratp",
    r"development bank", r"university", r"universit[ée]", r"hospital\w*", r"h[ôo]pitaux",
    # securitisations and funding vehicles
    # (generic words like "funding" or "securities" are not used: listed groups
    # such as Land Securities or Kingspan issue through vehicles with those names)
    r"clo", r"rmbs", r"securitisation", r"exmoor funding", r"pacific life global funding",
    r"pepper residential securities", r"finsbury square", r"green storm",
    r"pass through trust", r"sukuk", r"trust \d+",
]
_NON_LISTED_RE = re.compile(r"\b(" + "|".join(_NON_LISTED_PATTERNS) + r")\b")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_bond_list(source) -> pd.DataFrame:
    """Read the Euronext ESG bond Excel (or a CSV with the same columns).

    ``source`` is a path or a file-like object. The function finds the header
    row by itself, renames columns to standard names and parses dates and
    numbers. Columns it cannot find are created empty.
    """
    name = getattr(source, "name", str(source)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(source, header=None, dtype=str)
    else:
        raw = pd.read_excel(source, header=None, dtype=str)

    # header row = first row that contains an "ISIN" cell
    header_row = None
    for i in range(min(len(raw), 40)):
        if any(_normalise(c) in ("isin", "isin code") for c in raw.iloc[i].dropna()):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find a header row with an 'ISIN' column.")

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
    """Accept ISO dates (2012-03-27) from the Excel file and 27/03/2012 from the website."""
    iso = pd.to_datetime(col, errors="coerce", format="ISO8601")
    dmy = pd.to_datetime(col, errors="coerce", format="%d/%m/%Y")
    return iso.fillna(dmy)


def _parse_number(x) -> float:
    """Parse 1400, '1 000,00' (European) or '1,000.00' (English)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace("\u00a0", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) != 3 else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def is_public_issuer(name: str) -> bool:
    """True for governments, public agencies, supranationals and securitisation vehicles."""
    return bool(_NON_LISTED_RE.search(_normalise(name)))


def filter_bonds(bonds: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Apply the sample filters from the settings and flag public issuers."""
    b = bonds.copy()
    b["is_public"] = b["issuer"].map(is_public_issuer)
    wanted = {t.lower() for t in s.bond_types}
    # "Green Bond" -> "green"; "Sustainability-Linked Bond" -> "sustainability-linked"
    b["type_short"] = b["type"].str.lower().str.replace(" bonds", "").str.replace(" bond", "").str.strip()
    keep = b["type_short"].isin(wanted)
    keep &= b["listing_date"].between(pd.Timestamp(s.start), pd.Timestamp(s.end))
    keep &= b["nominal_mm"].fillna(0) >= s.min_nominal_mm
    if s.exclude_public_issuers:
        keep &= ~b["is_public"]
    return b[keep].reset_index(drop=True)


def first_issues(bonds: pd.DataFrame) -> pd.DataFrame:
    """One row per issuer: date of its first bond in the sample, count and total size."""
    g = bonds.sort_values("listing_date").groupby("issuer", as_index=False)
    return g.agg(event_date=("listing_date", "first"), first_isin=("isin", "first"),
                 n_bonds=("isin", "size"), total_nominal_mm=("nominal_mm", "sum"))


# =============================================================================
# 3  Matching issuers to stock tickers
# =============================================================================

# legal forms and filler words, matched after dots are removed ("S.p.A." -> "spa")
_LEGAL_FORMS = r"\b(sa|se|nv|plc|ag|ab|asa|as|oyj|spa|ltd|limited|group|groupe|holding|holdings|" \
               r"finance|financing|finco|treasury|international|entities|bv|gmbh|dac|sarl|sas|" \
               r"corporation|corp|company|co|inc|llc|the|de|du|la|le|a/s|a-s|hf|pjsc|tas|aş)\b"

# Yahoo exchange codes: primary European listings get a bonus, secondary German
# venues and over-the-counter quotes a penalty, so GFC.PA beats GI6A.F.
_PRIMARY_EXCHANGES = {"PAR", "AMS", "BRU", "LIS", "MIL", "MCE", "STO", "OSL", "CPH", "HEL", "ICE",
                      "GER", "LSE", "VIE", "EBS", "IST", "ATH", "ISE", "WSE", "PRA", "BUD", "NYQ", "NMS"}
_SECONDARY_EXCHANGES = {"FRA", "DUS", "MUN", "STU", "BER", "HAM", "HAN", "PNK", "OQX", "OQB"}


def _strip_accents(s: str) -> str:
    import unicodedata
    s = s.replace("ø", "o").replace("Ø", "O").replace("æ", "ae").replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def clean_company_name(name: str) -> str:
    n = _strip_accents(_normalise(name))
    n = n.replace("société anonyme", " ").replace("societe anonyme", " ")
    n = re.sub(r"\(.*?\)", " ", n)                 # "(LA)", "(COMPAGNIE DE)"
    n = n.replace(".", "")
    n = re.sub(r"[^\w\s&'/-]", " ", n)
    n = re.sub(_LEGAL_FORMS, " ", n)
    return re.sub(r"\s+", " ", n).strip()


def name_similarity(a: str, b: str) -> float:
    """0-1 similarity of two cleaned names; full credit when all words of the
    shorter name appear in the longer one ("gecina" vs "gecina sa")."""
    a, b = clean_company_name(a), clean_company_name(b)
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    contained = len(small & big) / len(small)
    return max(ratio, 0.95 * contained if contained == 1.0 else 0.8 * contained * ratio + 0.2 * ratio)


_BOND_LIKE = re.compile(r"%|\d+/\d+|^[A-Z]{2}[A-Z0-9]{9}\d")  # coupons, maturities, ISIN symbols


def score_quote(issuer: str, q: dict) -> float:
    """Score one Yahoo search result as the stock of ``issuer`` (0-1)."""
    if q.get("quoteType") != "EQUITY":
        return 0.0
    qname = q.get("longname") or q.get("shortname") or ""
    if _BOND_LIKE.search(qname) or _BOND_LIKE.search(q.get("symbol", "")):
        return 0.0                                   # an exchange-listed bond, not a share
    a, b = clean_company_name(issuer), clean_company_name(qname)
    s = name_similarity(issuer, qname)
    # short names ("bkk", "astm", "nge") must appear as a whole word
    if len(a) <= 5 and a not in b.split():
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
    """Search Yahoo Finance for each issuer and propose a stock ticker.

    Returns issuer, ticker, quote_name, exchange, score (0-1 name similarity)
    and include (True when the score is high). Results are cached in
    ``cache_path`` so a second run does not search again. Always review rows
    with a low score by hand before using them.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    rows = []
    for i, issuer in enumerate(issuers):
        if progress:
            progress(i, len(issuers), issuer)
        query = clean_company_name(issuer) or issuer
        quotes = cache.get(issuer)
        if quotes is None:
            quotes = search(query)
            cache[issuer] = quotes
        best, best_score = None, 0.0
        for q in quotes:
            sc = score_quote(issuer, q)
            if sc > best_score:
                best, best_score = q, sc
        rows.append({
            "issuer": issuer,
            "ticker": best.get("symbol") if best else "",
            "quote_name": (best.get("longname") or best.get("shortname")) if best else "",
            "exchange": best.get("exchange", "") if best else "",
            "score": round(best_score, 2),
            "include": bool(best) and best_score >= 0.75,
        })
    if cache_path:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return pd.DataFrame(rows)


# =============================================================================
# 4  Market data (prices and Fama-French factors)
# =============================================================================

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRENCH_FILES = {
    ("Europe", "monthly"): "Europe_5_Factors",
    ("Europe", "daily"): "Europe_5_Factors_Daily",
    ("Developed", "monthly"): "Developed_5_Factors",
    ("Developed", "daily"): "Developed_5_Factors_Daily",
}


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily adjusted close prices from Yahoo Finance, one column per ticker."""
    import yfinance as yf
    tickers = sorted({t for t in tickers if isinstance(t, str) and t})
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False, threads=True)
    if data is None or data.empty:
        raise ConnectionError("Yahoo Finance returned no data. Check your internet "
                              "connection (a VPN may be needed) or upload a prices CSV.")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if not isinstance(data.columns, pd.MultiIndex):
        close.columns = tickers
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all", axis=1)


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Kenneth French factor CSV (monthly YYYYMM or daily YYYYMMDD rows).

    Keeps only the first block (the annual block below it is ignored) and
    converts percent to decimals.
    """
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
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    fmt = "%Y%m" if len(df["date"].iloc[0]) == 6 else "%Y%m%d"
    idx = pd.to_datetime(df["date"], format=fmt)
    if fmt == "%Y%m":
        idx = idx + pd.offsets.MonthEnd(0)
    df = df.drop(columns="date").astype(float) / 100.0
    df.index = idx
    df.index.name = "date"
    return df


def download_french_factors(region: str = "Europe", freq: str = "monthly") -> pd.DataFrame:
    import requests
    name = FRENCH_FILES[(region, freq)]
    r = requests.get(FRENCH_URL.format(name=name), timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_french_csv(text)


def read_french_file(path) -> pd.DataFrame:
    """Read a Kenneth French factor file saved locally (.zip as downloaded, or the .csv inside)."""
    path = str(path)
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        text = z.read(z.namelist()[0]).decode("latin-1")
    else:
        text = open(path, encoding="latin-1").read()
    return parse_french_csv(text)


def read_prices_csv(source) -> pd.DataFrame:
    """Prices CSV: first column = date, other columns = tickers."""
    df = pd.read_csv(source, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# =============================================================================
# 5  Returns, winsorising and matched control firms
# =============================================================================

def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    m = prices.resample("ME").last()
    return m.pct_change(fill_method=None).iloc[1:]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).iloc[1:]


def winsorise(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Clip every cross-section (row) at the pct and 100-pct percentiles."""
    if pct <= 0:
        return df
    lo = df.quantile(pct / 100, axis=1)
    hi = df.quantile(1 - pct / 100, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def _beta_vol(window: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """Market beta, volatility and number of months for every column, in one pass.

    Each column uses only the months where both it and the market are
    observed.
    """
    R = window.to_numpy(dtype=float)
    m = mkt.reindex(window.index).to_numpy(dtype=float)[:, None]
    ok = ~np.isnan(R) & ~np.isnan(m)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Rz, Mz = np.where(ok, R, 0.0), np.where(ok, m, 0.0)
        r_bar, m_bar = Rz.sum(0) / n, Mz.sum(0) / n
        dr, dm = np.where(ok, R - r_bar, 0.0), np.where(ok, m - m_bar, 0.0)
        cov = (dr * dm).sum(0) / (n - 1)
        var_m = (dm ** 2).sum(0) / (n - 1)
        vol = np.sqrt((dr ** 2).sum(0) / (n - 1))
        beta = cov / var_m
    return pd.DataFrame({"beta": beta, "vol": vol, "n": n}, index=window.columns)


def match_controls(events: pd.DataFrame, rets_m: pd.DataFrame, mkt_m: pd.Series,
                   universe: list[str], s: Settings) -> pd.DataFrame:
    """Pick one non-issuer control firm per issuer.

    For every issuer, beta and volatility are measured over the
    ``match_window_months`` before the event, for the issuer and for every
    candidate in ``universe`` that is not an issuer. The control is the
    candidate with the smallest standardised distance in (beta, volatility).
    This is a simple, transparent rule, chosen for this project.
    """
    issuers = set(events["ticker"])
    candidates = [t for t in universe if t in rets_m.columns and t not in issuers]
    used, rows = set(), []
    for _, e in events.sort_values("event_date").iterrows():
        t0 = pd.Timestamp(e["event_date"]) + pd.offsets.MonthEnd(0)
        window = rets_m.loc[(rets_m.index < t0) &
                            (rets_m.index >= t0 - pd.DateOffset(months=s.match_window_months))]
        empty = {"ticker": e["ticker"], "control": None, "distance": np.nan}
        if e["ticker"] not in window or window[e["ticker"]].count() < s.match_min_months:
            rows.append(empty)
            continue
        own = _beta_vol(window[[e["ticker"]]], mkt_m).iloc[0]
        pool = [c for c in candidates if s.match_with_replacement or c not in used]
        st = _beta_vol(window[pool], mkt_m) if pool else pd.DataFrame()
        st = st[st["n"] >= s.match_min_months].dropna() if not st.empty else st
        if st.empty:
            rows.append(empty)
            continue
        sb, sv = st["beta"].std() or 1.0, st["vol"].std() or 1.0
        d = np.sqrt(((st["beta"] - own["beta"]) / sb) ** 2 + ((st["vol"] - own["vol"]) / sv) ** 2)
        c = d.idxmin()
        used.add(c)
        rows.append({"ticker": e["ticker"], "control": c, "distance": float(d[c]),
                     "beta_issuer": own["beta"], "beta_control": st.loc[c, "beta"],
                     "vol_issuer": own["vol"], "vol_control": st.loc[c, "vol"]})
    return pd.DataFrame(rows)


# =============================================================================
# 6  Calendar-time portfolios and factor regressions
# =============================================================================

def holding_mask(events: pd.DataFrame, index: pd.DatetimeIndex, s: Settings,
                 key: str = "ticker") -> pd.DataFrame:
    """0/1 matrix (months x firms): 1 while the firm is held in the portfolio."""
    cols = [t for t in events[key].dropna().unique()]
    mask = pd.DataFrame(0, index=index, columns=cols, dtype=float)
    for _, e in events.dropna(subset=[key]).iterrows():
        # month-end of the listing month, then move forward entry_lag_months
        listing_month = pd.Timestamp(e["event_date"]) + pd.offsets.MonthEnd(0)
        first = listing_month + pd.offsets.MonthEnd(s.entry_lag_months)
        last = first + pd.offsets.MonthEnd(s.hold_months - 1) if s.hold_months > 0 else index.max()
        mask.loc[(index >= first) & (index <= last), e[key]] = 1.0
    return mask


def portfolio_returns(rets_m: pd.DataFrame, mask: pd.DataFrame, min_firms: int) -> pd.DataFrame:
    """Equal-weighted portfolio return and number of firms, month by month."""
    cols = [c for c in mask.columns if c in rets_m.columns]
    r = rets_m.reindex(mask.index)[cols]
    held = mask[cols].where(r.notna(), 0.0)
    n = held.sum(axis=1)
    ret = (r.fillna(0.0) * held).sum(axis=1) / n.replace(0, np.nan)
    out = pd.DataFrame({"ret": ret, "n_firms": n})
    return out[out["n_firms"] >= min_firms]


def factor_regression(excess: pd.Series, factors: pd.DataFrame, model: str,
                      nw_lags: int = 6) -> dict:
    """Alpha and loadings of a monthly excess return on a factor model.

    Standard errors are Newey-West with ``nw_lags`` lags. Alpha is reported
    per month and annualised (x12), both in percent.
    """
    cols = FACTOR_MODELS[model]
    data = pd.concat([excess.rename("y"), factors[cols]], axis=1, join="inner", sort=False).dropna()
    if len(data) < len(cols) + 12:
        return {"model": model, "n_months": len(data), "error": "too few months"}
    X = sm.add_constant(data[cols])
    res = sm.OLS(data["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    out = {"model": model, "n_months": int(len(data)),
           "alpha_m_pct": 100 * res.params["const"], "alpha_ann_pct": 1200 * res.params["const"],
           "t_alpha": res.tvalues["const"], "p_alpha": res.pvalues["const"],
           "r2": res.rsquared}
    for c in cols:
        out[f"b_{c}"] = res.params[c]
        out[f"t_{c}"] = res.tvalues[c]
    return out


def performance_summary(ret: pd.Series, rf: pd.Series | None = None) -> dict:
    r = ret.dropna()
    if len(r) < 2:
        return {"months": len(r)}
    ex = r - rf.reindex(r.index).fillna(0) if rf is not None else r
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1 if len(r) else np.nan
    vol = r.std() * np.sqrt(12)
    return {"months": len(r), "mean_m_pct": 100 * r.mean(), "ann_return_pct": 100 * ann_ret,
            "ann_vol_pct": 100 * vol,
            "sharpe": (ex.mean() * 12) / vol if vol > 0 else np.nan,
            "max_drawdown_pct": 100 * ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()}


# =============================================================================
# 7  Event study
# =============================================================================

def event_study(events: pd.DataFrame, rets_d: pd.DataFrame, mkt_d: pd.Series,
                s: Settings, path_window: tuple = (-10, 10)) -> dict:
    """Market-model event study around each issuer's first green bond listing.

    For each event, alpha and beta are estimated over ``est_window`` trading
    days; abnormal returns are actual minus predicted returns. Returns one row
    per event with the CAR for every window in ``event_windows``, a summary
    table (mean CAR, cross-sectional t-statistic, share positive), and the
    average cumulative abnormal return path for plotting.
    """
    dates = rets_d.index
    lo_all = min(s.est_window[0], path_window[0], *(w[0] for w in s.event_windows))
    hi_all = max(path_window[1], *(w[1] for w in s.event_windows))
    rows, paths = [], []
    mkt = mkt_d.reindex(dates)
    for _, e in events.dropna(subset=["ticker"]).iterrows():
        if e["ticker"] not in rets_d:
            continue
        pos = dates.searchsorted(pd.Timestamp(e["event_date"]))  # first trading day >= listing
        if pos + lo_all < 0 or pos + hi_all >= len(dates):
            continue
        r = rets_d[e["ticker"]].to_numpy()
        m = mkt.to_numpy()
        est = slice(pos + s.est_window[0], pos + s.est_window[1] + 1)
        ok = ~np.isnan(r[est]) & ~np.isnan(m[est])
        if ok.sum() < s.event_min_est_days:
            continue
        b, a = np.polyfit(m[est][ok], r[est][ok], 1)
        sigma = np.std(r[est][ok] - (a + b * m[est][ok]), ddof=2)
        rel = np.arange(lo_all, hi_all + 1)
        ar = pd.Series(r[pos + rel] - (a + b * m[pos + rel]), index=rel)
        row = {"issuer": e["issuer"], "ticker": e["ticker"], "event_date": e["event_date"],
               "beta": b, "sigma_est": sigma}
        for w in s.event_windows:
            row[f"CAR[{w[0]},{w[1]}]"] = ar.loc[w[0]:w[1]].sum()
        rows.append(row)
        paths.append(ar.loc[path_window[0]:path_window[1]])
    per_event = pd.DataFrame(rows)
    summary = []
    for w in s.event_windows:
        col = f"CAR[{w[0]},{w[1]}]"
        if per_event.empty:
            break
        x = per_event[col].dropna()
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan
        summary.append({"window": col, "n_events": len(x), "mean_CAR_pct": 100 * x.mean(),
                        "median_CAR_pct": 100 * x.median(), "t_stat": t,
                        "share_positive": (x > 0).mean()})
    caar = (pd.concat(paths, axis=1, sort=False).mean(axis=1).cumsum() if paths else pd.Series(dtype=float))
    return {"per_event": per_event, "summary": pd.DataFrame(summary), "caar_path": caar}


# =============================================================================
# 8  Full pipeline and robustness grid
# =============================================================================

def run_analysis(events: pd.DataFrame, prices: pd.DataFrame, factors_m: pd.DataFrame,
                 factors_d: pd.DataFrame | None, universe: list[str], s: Settings) -> dict:
    """Run portfolios, regressions and the event study.

    ``events`` needs columns issuer, ticker, event_date (one row per issuer,
    already restricted to the firms you want to include).
    """
    events = events.dropna(subset=["ticker"])
    events = events[events["ticker"].isin(prices.columns)].copy()
    # one firm can appear under several issuer names (e.g. "BBVA" and "Banco Bilbao
    # Vizcaya Argentaria"); keep its earliest event only
    events = events.sort_values("event_date").drop_duplicates("ticker").reset_index(drop=True)
    rets_m = winsorise(monthly_returns(prices), s.winsor_pct)
    fm = factors_m.reindex(rets_m.index)
    mkt_m = fm["Mkt-RF"] + fm["RF"]

    matches = match_controls(events, rets_m, mkt_m, universe, s)
    events = events.merge(matches[["ticker", "control"]], on="ticker", how="left")

    idx = rets_m.index
    green = portfolio_returns(rets_m, holding_mask(events, idx, s, "ticker"), s.min_firms)
    matched = events.dropna(subset=["control"])
    control = (portfolio_returns(rets_m, holding_mask(matched, idx, s, "control"), s.min_firms)
               if not matched.empty else pd.DataFrame(columns=["ret", "n_firms"], dtype=float))
    rf = fm["RF"]
    ports = pd.DataFrame({"green": green["ret"], "control": control["ret"]}).dropna(how="all")
    ports = ports.loc[pd.Timestamp(s.start):pd.Timestamp(s.end) + pd.offsets.MonthEnd(s.hold_months + 1)]
    ports = ports.loc[:factors_m.index.max()]   # stop where the factor data ends (no partial months)
    ports["green_minus_control"] = ports["green"] - ports["control"]
    ports["n_green"] = green["n_firms"]
    ports["n_control"] = control["n_firms"]

    regs = []
    for name in ("green", "control", "green_minus_control"):
        y = ports[name].dropna()
        if y.empty:
            continue
        y_ex = y if name == "green_minus_control" else y - rf.reindex(y.index)
        for model in FACTOR_MODELS:
            regs.append({"portfolio": name, **factor_regression(y_ex, fm, model, s.nw_lags)})

    # the long-short portfolio is already an excess return, so no risk-free rate there
    perf = pd.DataFrame([
        {"portfolio": n, **performance_summary(ports[n], None if n == "green_minus_control" else rf)}
        for n in ("green", "control", "green_minus_control")])

    es = None
    if factors_d is not None and not factors_d.empty:
        rets_d = winsorise(daily_returns(prices), 0)
        mkt_d = (factors_d["Mkt-RF"] + factors_d["RF"]).reindex(rets_d.index)
        es = event_study(events, rets_d, mkt_d, s)

    return {"events": events, "matches": matches, "portfolios": ports,
            "regressions": pd.DataFrame(regs), "performance": perf, "event_study": es,
            "settings": s}


def robustness_grid(events, prices, factors_m, universe, s: Settings,
                    hold_list=(12, 24, 36, 60), winsor_list=(0.0, 1.0, 5.0),
                    models=None) -> pd.DataFrame:
    """Green-minus-control alpha for every holding period x winsorising level.

    ``models`` lists the factor models to keep (default: only s.factor_model).
    The output has a ``model`` column, so one grid can hold all three.
    """
    from dataclasses import replace
    models = tuple(models or (s.factor_model,))
    rows = []
    for h in hold_list:
        for w in winsor_list:
            res = run_analysis(events, prices, factors_m, None, universe,
                               replace(s, hold_months=h, winsor_pct=w))
            r = res["regressions"]
            for m in models:
                rm = r[(r.portfolio == "green_minus_control") & (r.model == m)]
                if not rm.empty and "alpha_ann_pct" in rm and pd.notna(rm["alpha_ann_pct"].iloc[0]):
                    rows.append({"model": m, "hold_months": h, "winsor_pct": w,
                                 "alpha_ann_pct": rm["alpha_ann_pct"].iloc[0],
                                 "t_alpha": rm["t_alpha"].iloc[0], "n_months": rm["n_months"].iloc[0]})
    return pd.DataFrame(rows)


# =============================================================================
# 9  Demo data (works offline)
# =============================================================================

def make_demo_data(seed: int = 7, n_issuers: int = 120, n_universe: int = 300,
                   green_alpha_ann: float = 0.02, announce_effect: float = 0.004,
                   idio_vol_daily: float = 0.012,
                   start: str = "2010-01-01", end: str = "2025-12-31") -> dict:
    """Synthetic data in the same formats as the real sources.

    The demo exists so the dashboard and tests run without internet. The true
    values are known: issuers earn ``green_alpha_ann`` extra per year after
    their first listing, and ``announce_effect`` on the listing day.
    ``idio_vol_daily`` is the average firm-specific daily volatility; real
    European stocks are closer to 0.015-0.02, which makes small effects hard
    to detect. These are made-up numbers for checking the code, not results.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    T = len(days)
    # daily factors (decimals)
    fac = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0003, 0.011, T), "SMB": rng.normal(0.0, 0.005, T),
        "HML": rng.normal(0.0001, 0.005, T), "RMW": rng.normal(0.0001, 0.004, T),
        "CMA": rng.normal(0.0, 0.004, T), "RF": np.full(T, 0.00004),
    }, index=days)

    n = n_issuers + n_universe
    tickers = [f"GB{i:03d}.DEMO" for i in range(n_issuers)] + \
              [f"CT{i:03d}.DEMO" for i in range(n_universe)]
    load = np.column_stack([rng.normal(1.0, 0.25, n), rng.normal(0.2, 0.4, n),
                            rng.normal(0.1, 0.4, n), rng.normal(0.0, 0.3, n),
                            rng.normal(0.0, 0.3, n)])
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
    mapping = pd.DataFrame({"issuer": names, "ticker": tickers[:n_issuers],
                            "quote_name": names, "exchange": "DEMO", "score": 1.0, "include": True})
    factors_m = (1 + fac).resample("ME").prod() - 1
    return {"bonds": bonds, "prices": prices, "factors_d": fac, "factors_m": factors_m,
            "mapping": mapping, "universe": tickers[n_issuers:],
            "true": {"green_alpha_ann_pct": 100 * green_alpha_ann,
                     "announce_effect_pct": 100 * announce_effect}}


def bonds_to_excel_bytes(bonds_raw: pd.DataFrame) -> bytes:
    """Write a bond table in Euronext-like layout (title rows, then header)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([["ESG Bonds list (demo)"], [""]]).to_excel(xw, index=False, header=False)
        bonds_raw.to_excel(xw, index=False, startrow=2)
    return buf.getvalue()


def demo_events(demo: dict, s: Settings) -> pd.DataFrame:
    b = filter_bonds(load_bond_list(io.BytesIO(bonds_to_excel_bytes(demo["bonds"]))), s)
    ev = first_issues(b).merge(demo["mapping"][["issuer", "ticker"]], on="issuer", how="left")
    return ev


# =============================================================================
# 9b Saved results (a shareable snapshot without the price data)
# =============================================================================
# Yahoo prices cannot be republished, so a public copy of the project cannot
# reproduce the real-data results from scratch. save_snapshot() stores the
# results of a run, without prices, in a JSON file for the dashboard to display.

def _frame_to_obj(df) -> dict:
    """DataFrame or Series -> plain JSON-ready dict that keeps dates and NaN."""
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
    for c in df.columns:     # columns of None/numbers come back as object
        if df[c].dtype == object and c not in o["date_columns"]:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df


def save_snapshot(res: dict, s: Settings, path, grid: pd.DataFrame | None = None,
                  issuers: pd.DataFrame | None = None, meta: dict | None = None) -> None:
    """Write the results of one run (no prices) to a JSON file."""
    es = res["event_study"]
    frames = {"events": res["events"], "matches": res["matches"], "portfolios": res["portfolios"],
              "regressions": res["regressions"], "performance": res["performance"]}
    if es is not None:
        frames.update(es_per_event=es["per_event"], es_summary=es["summary"],
                      es_caar_path=es["caar_path"].rename("caar"))
    if grid is not None:
        frames["grid"] = grid
    if issuers is not None:
        frames["issuers"] = issuers[[c for c in ("issuer", "event_date", "n_bonds") if c in issuers]]
    out = {"format": "greenbond-snapshot-1", "settings": asdict(s), "meta": meta or {},
           "frames": {k: _frame_to_obj(v) for k, v in frames.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), default=str)


def load_snapshot(path) -> dict:
    """Read a file written by save_snapshot(). Returns res (as run_analysis), grid, issuers, meta."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    fr = {k: _obj_to_frame(v) for k, v in raw["frames"].items()}
    st = dict(raw["settings"])
    st["bond_types"] = tuple(st["bond_types"])
    st["est_window"] = tuple(st["est_window"])
    st["event_windows"] = tuple(tuple(w) for w in st["event_windows"])
    s = Settings(**st)
    es = None
    if "es_summary" in fr:
        path_ = fr["es_caar_path"]["caar"]
        path_.index = path_.index.astype(int)
        es = {"per_event": fr["es_per_event"], "summary": fr["es_summary"], "caar_path": path_}
    res = {"events": fr["events"], "matches": fr["matches"], "portfolios": fr["portfolios"],
           "regressions": fr["regressions"], "performance": fr["performance"],
           "event_study": es, "settings": s}
    return {"res": res, "settings": s, "grid": fr.get("grid"), "issuers": fr.get("issuers"),
            "meta": raw.get("meta", {})}


def build_snapshot(data_dir, out_path, s: Settings | None = None, region: str = "Europe") -> dict:
    """Run the full analysis on the files in data_dir and save a snapshot.

    Needs, in data_dir: the Euronext bond list, issuer_tickers.csv,
    control_universe.csv, prices.csv(.gz) and the Kenneth French factor zips.
    """
    s = s or Settings()
    d = os.fspath(data_dir)

    def pick(*names):
        for n in names:
            if os.path.exists(os.path.join(d, n)):
                return os.path.join(d, n)
        raise FileNotFoundError(f"none of {names} in {d}")

    bonds_all = load_bond_list(pick("Euronext-Green-Bond-List.xlsx", "esg_bonds.xlsx", "esg_bonds.csv"))
    bonds = filter_bonds(bonds_all, s)
    issuers = first_issues(bonds)
    mapping = pd.read_csv(pick("issuer_tickers.csv"))
    mapping = mapping[mapping["include"].astype(str).str.lower().isin(["true", "1", "yes"])]
    events = issuers.merge(mapping[["issuer", "ticker"]], on="issuer", how="left")
    universe = pd.read_csv(pick("control_universe.csv"))["ticker"].astype(str).tolist()
    prices = read_prices_csv(pick("prices.csv.gz", "prices.csv"))
    fm = read_french_file(pick(f"{region}_5_Factors_CSV.zip", f"{region}_5_Factors.csv"))
    fd = read_french_file(pick(f"{region}_5_Factors_Daily_CSV.zip", f"{region}_5_Factors_Daily.csv"))

    res = run_analysis(events, prices, fm, fd, universe, s)
    grid = robustness_grid(events, prices, fm, universe, s, hold_list=(12, 24, 36, 60),
                           winsor_list=(0.0, 1.0, 5.0), models=tuple(FACTOR_MODELS))
    from dataclasses import replace
    all_types = filter_bonds(bonds_all, replace(s, exclude_public_issuers=False))
    meta = {"computed_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "prices_to": prices.dropna(how="all").index.max().strftime("%Y-%m-%d"),
            "factors_to": fm.index.max().strftime("%Y-%m-%d"),
            "factor_region": region,
            "bonds_in_file": int(len(bonds_all)),
            "bonds_after_filters": int(len(bonds)),
            "public_excluded": int(all_types["is_public"].sum()) if s.exclude_public_issuers else 0,
            "issuers": int(len(issuers)),
            "control_universe": int(len(universe))}
    save_snapshot(res, s, out_path, grid=grid, issuers=issuers, meta=meta)
    return res


# =============================================================================
# 10 Command line
# =============================================================================

def _print_results(res: dict):
    pd.set_option("display.width", 160)
    r = res["regressions"]
    cols = ["portfolio", "model", "n_months", "alpha_m_pct", "alpha_ann_pct", "t_alpha", "r2"]
    print("\nFactor regressions")
    print(r[cols].round(3).to_string(index=False))
    if res["event_study"] is not None:
        print("\nEvent study")
        print(res["event_study"]["summary"].round(3).to_string(index=False))


def selftest():
    """Quick checks of every building block, run with:  python greenbond_engine.py --selftest"""
    checks = []

    def check(name, ok):
        checks.append(ok)
        print(("PASS " if ok else "FAIL ") + name)

    # parsing
    check("European number format", _parse_number("1 000,00") == 1000.0 and _parse_number("350,00") == 350.0)
    check("English number format", _parse_number("1,000.50") == 1000.5 and _parse_number(1400) == 1400.0)
    check("public issuers flagged", all(map(is_public_issuer, ["EIB", "REPUBLIC OF ITALY", "VILLE DE PARIS",
                                                               "REGION WALLONNE", "KFW"])))
    check("listed groups kept", not any(map(is_public_issuer, ["LAND SECURITIES", "GECINA", "ENEL FINANCE "
                                                                "INTERNATIONAL", "SBB TREASURY OYJ"])))
    check("company names cleaned", clean_company_name("A2A S.p.A.") == "a2a"
          and clean_company_name("SAINT-GOBAIN (COMPAGNIE DE)") == "saint-gobain")
    q = [{"symbol": "GI6A.F", "longname": "Gecina S.A.", "exchange": "FRA", "quoteType": "EQUITY"},
         {"symbol": "GFC.PA", "longname": "Gecina", "exchange": "PAR", "quoteType": "EQUITY"}]
    check("primary listing preferred", suggest_tickers(["GECINA"], search=lambda _: q)["ticker"][0] == "GFC.PA")
    text = "header\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n199007,1.0,2.0,3.0,4.0,5.0,0.5\n\n Annual\n"
    f = parse_french_csv(text)
    check("French factor file parsed", f.index[0] == pd.Timestamp("1990-07-31") and f.iloc[0, 0] == 0.01)

    # estimators recover known effects on low-noise demo data
    alphas, cars = [], []
    for seed in range(1, 5):
        d = make_demo_data(seed=seed, idio_vol_daily=0.002)
        st_ = Settings(start="2012-01-01", winsor_pct=0)
        res = run_analysis(demo_events(d, st_), d["prices"], d["factors_m"], d["factors_d"],
                           d["universe"], st_)
        r = res["regressions"]
        alphas.append(r[(r.portfolio == "green_minus_control") & (r.model == "FF5")]["alpha_ann_pct"].iloc[0])
        cars.append(res["event_study"]["summary"]["mean_CAR_pct"].iloc[0])
    check(f"portfolio alpha recovered (mean {np.mean(alphas):.2f} vs true 2.00 % a year)",
          abs(np.mean(alphas) - 2.0) < 0.6)
    check(f"announcement effect recovered (mean {np.mean(cars):.2f} vs true 0.40 %)",
          abs(np.mean(cars) - 0.4) < 0.15)
    # saved results survive a round trip through the JSON file
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "snap.json")
        save_snapshot(res, st_, fn)
        back = load_snapshot(fn)["res"]
    check("saved results reload unchanged",
          np.allclose(back["regressions"]["alpha_ann_pct"], res["regressions"]["alpha_ann_pct"],
                      equal_nan=True)
          and back["portfolios"].index.equals(res["portfolios"].index)
          and np.allclose(back["event_study"]["caar_path"], res["event_study"]["caar_path"]))
    print(f"\n{sum(checks)} of {len(checks)} checks passed.")
    if not all(checks):
        raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Green bond issuer stock performance")
    ap.add_argument("--demo", action="store_true", help="run on synthetic demo data")
    ap.add_argument("--bonds", help="Euronext ESG bond list (.xlsx or .csv)")
    ap.add_argument("--suggest-tickers", metavar="OUT_CSV",
                    help="search Yahoo for each issuer's ticker and save a CSV to review")
    ap.add_argument("--selftest", action="store_true",
                    help="check the building blocks and that known effects are recovered")
    ap.add_argument("--save-snapshot", metavar="OUT_JSON",
                    help="run on the files in --data-dir with default settings and save the "
                         "results (no prices) for the dashboard's 'Saved results' view")
    ap.add_argument("--data-dir", default="data", help="folder with the data files (default: data)")
    args = ap.parse_args(argv)
    s = Settings()

    if args.save_snapshot:
        res = build_snapshot(args.data_dir, args.save_snapshot, s)
        _print_results(res)
        print(f"\nSaved {args.save_snapshot}")
        return

    if args.selftest:
        selftest()
        return

    if args.demo:
        demo = make_demo_data()
        s.start = "2012-01-01"
        res = run_analysis(demo_events(demo, s), demo["prices"], demo["factors_m"],
                           demo["factors_d"], demo["universe"], s)
        print("Demo data. True values:", demo["true"])
        _print_results(res)
        return

    if args.bonds and args.suggest_tickers:
        bonds = filter_bonds(load_bond_list(args.bonds), s)
        issuers = first_issues(bonds)["issuer"].tolist()
        print(f"{len(issuers)} issuers after filters. Searching Yahoo Finance...")
        out = suggest_tickers(issuers, cache_path="ticker_search_cache.json",
                              progress=lambda i, n, x: print(f"  {i + 1}/{n} {x}"))
        out.to_csv(args.suggest_tickers, index=False)
        print(f"Saved {args.suggest_tickers}. Review rows with include = False or a low score.")
        return
    ap.print_help()


if __name__ == "__main__":
    main()

