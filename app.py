"""
app.py - interactive dashboard for greenbond_engine.py

Run with:   streamlit run app.py

Every setting in the sidebar is passed to the engine, so results update when
you change a value. "Demo data" works offline; "My data" uses the Euronext
ESG bond list, Yahoo Finance prices and Kenneth French factors.
"Saved results" shows the real-data results stored in data/saved_results.json,
so the app can run online without the price data (which cannot be shared).

Author: Saeed Samadiye.
"""
from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import greenbond_engine as eng

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SNAPSHOT_FILE = DATA_DIR / "saved_results.json"


def first_existing(*names):
    """Path (as str) of the first of these files that exists in DATA_DIR, else None."""
    for n in names:
        if (DATA_DIR / n).exists():
            return str(DATA_DIR / n)
    return None

st.set_page_config(page_title="Green Bond Issuers", page_icon="🌿", layout="wide")

# Chart colours and neutral text colours
C_GREEN, C_CONTROL, C_DIFF, C_ACCENT = "#1baf7a", "#2a78d6", "#eb6834", "#eda100"
MUTED, GRID = "#6b6a64", "#e4e3de"


def style_fig(fig: go.Figure, height: int = 380, y_title: str = "", y_pct: bool = False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED, title=y_title,
                     tickformat=".1%" if y_pct else None)
    return fig


# --- Cached data ---
@st.cache_data(show_spinner="Building demo data...")
def cached_demo(seed, alpha, announce, vol):
    return eng.make_demo_data(seed=seed, green_alpha_ann=alpha / 100,
                              announce_effect=announce / 100, idio_vol_daily=vol / 100)


@st.cache_data(show_spinner="Reading the bond list...")
def cached_bonds(file_bytes: bytes, name: str):
    buf = io.BytesIO(file_bytes)
    buf.name = name
    return eng.load_bond_list(buf)


@st.cache_data(show_spinner="Downloading prices from Yahoo Finance...")
def cached_prices(tickers: tuple, start: str, end: str):
    return eng.download_prices(list(tickers), start, end)


@st.cache_data(show_spinner="Reading prices...")
def cached_prices_file(path: str):
    return eng.read_prices_csv(path)


@st.cache_data(show_spinner="Downloading Fama-French factors...")
def cached_factors(region: str, freq: str):
    return eng.download_french_factors(region, freq)


@st.cache_data(show_spinner="Loading saved results...")
def cached_snapshot(path: str, mtime: float):
    return eng.load_snapshot(path)


@st.cache_data(show_spinner="Running the analysis...")
def cached_analysis(events, prices, factors_m, factors_d, universe: tuple, settings_json: str):
    s = eng.Settings(**{k: tuple(map(tuple, v)) if k == "event_windows" else
                        (tuple(v) if isinstance(v, list) else v)
                        for k, v in __import__("json").loads(settings_json).items()})
    return eng.run_analysis(events, prices, factors_m, factors_d, list(universe), s)


@st.cache_data(show_spinner="Running the robustness grid...")
def cached_grid(events, prices, factors_m, universe: tuple, settings_json: str, holds, winsors):
    s = eng.Settings(**{k: tuple(map(tuple, v)) if k == "event_windows" else
                        (tuple(v) if isinstance(v, list) else v)
                        for k, v in __import__("json").loads(settings_json).items()})
    return eng.robustness_grid(events, prices, factors_m, list(universe), s, holds, winsors)


def parse_tickers(text: str) -> list[str]:
    return [t.strip() for t in text.replace(",", "\n").replace(";", "\n").splitlines() if t.strip()]


def parse_windows(text: str) -> tuple:
    out = []
    for part in text.split(";"):
        a, b = part.strip().strip("[]()").split(",")
        out.append((int(a), int(b)))
    return tuple(out)


# --- Sidebar settings ---
st.sidebar.title("🌿 Green Bond Issuers")
MODES = (["Saved results (real data)"] if SNAPSHOT_FILE.exists() else []) + \
        ["Demo data (offline)", "My data"]
mode = st.sidebar.radio("Data", MODES,
                        help="Saved results: real-data results computed with the default settings. "
                             "Demo data: synthetic, with known true effects. "
                             "My data: recompute everything from the files in data/ (needs prices).")
saved = mode.startswith("Saved")
snap = cached_snapshot(str(SNAPSHOT_FILE), SNAPSHOT_FILE.stat().st_mtime) if saved else None
# In "Saved results" mode, the settings are fixed to the values used for that run.
# The widgets remain visible but are locked. Model and significance level only
# affect the display, so those controls remain available.
S0 = snap["settings"] if saved else eng.Settings()
lock = saved

sb = st.sidebar
if saved:
    sb.caption("Settings are locked to the saved run. Switch to *My data* to change them.")
sb.header("Sample")
c1, c2 = sb.columns(2)
start = c1.date_input("Start", pd.Timestamp(S0.start), disabled=lock)
end = c2.date_input("End", pd.Timestamp(S0.end), disabled=lock)
bond_types = sb.multiselect("Bond types counted as events", list(eng.BOND_TYPES), list(S0.bond_types),
                            disabled=lock)
min_nominal = sb.number_input("Minimum bond size (millions)", 0.0, 10000.0, float(S0.min_nominal_mm),
                              50.0, disabled=lock)
excl_public = sb.checkbox("Exclude governments, agencies, supranationals", S0.exclude_public_issuers,
                          disabled=lock)

sb.header("Portfolios")
hold = sb.slider("Holding period after first green bond (months)", 6, 120, S0.hold_months, 6,
                 disabled=lock)
lag = sb.number_input("Start holding N months after the listing month", 0, 12, S0.entry_lag_months,
                      disabled=lock)
min_firms = sb.number_input("Drop months with fewer firms than", 1, 50, S0.min_firms, disabled=lock)
winsor = sb.select_slider("Winsorise monthly returns (each tail, %)", [0.0, 0.5, 1.0, 2.5, 5.0],
                          float(S0.winsor_pct), disabled=lock)

sb.header("Matching")
match_win = sb.slider("Pre-event window for beta and volatility (months)", 12, 60,
                      S0.match_window_months, 6, disabled=lock)
match_min = sb.number_input("Minimum months of data to match", 6, 60, S0.match_min_months, disabled=lock)
replacement = sb.checkbox("Allow one control firm to match several issuers", S0.match_with_replacement,
                          disabled=lock)

sb.header("Regressions")
model = sb.selectbox("Main factor model", list(eng.FACTOR_MODELS),
                     index=list(eng.FACTOR_MODELS).index(S0.factor_model))
nw = sb.number_input("Newey-West lags", 0, 24, S0.nw_lags, disabled=lock)
alpha_level = sb.select_slider("Significance level", [0.01, 0.05, 0.10], 0.05)

sb.header("Event study")
est = sb.slider("Estimation window (trading days)", -400, -10, tuple(S0.est_window), disabled=lock)
win_text = sb.text_input("Event windows (a,b; c,d)", "; ".join(f"{a},{b}" for a, b in S0.event_windows),
                         disabled=lock)
try:
    windows = parse_windows(win_text)
except Exception:
    sb.error("Write windows like  -1,1; 0,5")
    windows = ((-1, 1),)

settings = eng.Settings(
    start=str(start), end=str(end), bond_types=tuple(bond_types) or ("Green",),
    min_nominal_mm=float(min_nominal), exclude_public_issuers=excl_public,
    hold_months=int(hold), entry_lag_months=int(lag), min_firms=int(min_firms),
    winsor_pct=float(winsor), match_window_months=int(match_win),
    match_min_months=int(match_min), match_with_replacement=replacement,
    factor_model=model, nw_lags=int(nw), est_window=tuple(est), event_windows=windows)

# --- Load and prepare data ---
events = prices = factors_m = factors_d = None
universe: list[str] = []
bonds_all = bonds = None

if saved:
    meta = snap["meta"]
    res = snap["res"]
    issuers = snap["issuers"]
    mapping = (pd.read_csv(first_existing("issuer_tickers.csv"))
               if first_existing("issuer_tickers.csv") else None)
    universe = ["saved"]          # only used to tell the tabs that controls exist
    st.info(f"**Saved results (real data).** Computed on {meta.get('computed_on', '?')} with the "
            f"default settings shown in the sidebar: Euronext ESG bond list, Yahoo Finance prices "
            f"(to {meta.get('prices_to', '?')}) and Kenneth French {meta.get('factor_region', 'Europe')} "
            f"factors (to {meta.get('factors_to', '?')}). The price data is not included in this "
            "copy, so the settings are locked here. To change them, run the app locally with "
            "*My data* (it downloads prices from Yahoo Finance), or try *Demo data*.")
elif mode.startswith("Demo"):
    with st.sidebar.expander("Demo settings (true values)", expanded=False):
        d_alpha = st.slider("True extra return of issuers (% per year)", -5.0, 5.0, 2.0, 0.5)
        d_ann = st.slider("True listing-day reaction (%)", -1.0, 1.0, 0.4, 0.1)
        d_vol = st.slider("Firm-specific daily volatility (%)", 0.2, 2.5, 1.2, 0.1,
                          help="Higher = noisier, closer to real stocks; effects get harder to detect.")
        d_seed = st.number_input("Random seed", 0, 999, 7)
    demo = cached_demo(int(d_seed), d_alpha, d_ann, d_vol)
    bonds_all = eng.load_bond_list(io.BytesIO(eng.bonds_to_excel_bytes(demo["bonds"])))
    mapping = demo["mapping"]
    prices, factors_m, factors_d = demo["prices"], demo["factors_m"], demo["factors_d"]
    universe = demo["universe"]
    st.info(f"**Demo data.** Synthetic stocks with a true extra return of {d_alpha:.1f}% a year "
            f"after the first green bond and a true listing-day reaction of {d_ann:.1f}%. "
            "Use it to see how the method behaves; switch to *My data* for real results.")
else:
    # Files placed in the data/ folder next to app.py are used automatically;
    # anything uploaded in the sidebar replaces them.
    st.sidebar.header("My data")
    st.sidebar.caption(f"Files in `{DATA_DIR}` are loaded automatically. Upload a file to replace one.")

    # 1. bond list ------------------------------------------------------------------
    up = st.sidebar.file_uploader("1. Euronext ESG bond list (.xlsx or .csv)", ["xlsx", "csv"])
    local_bonds = first_existing("Euronext-Green-Bond-List.xlsx", "esg_bonds.xlsx", "esg_bonds.csv")
    if up is not None:
        bonds_all = cached_bonds(up.getvalue(), up.name)
    elif local_bonds:
        bonds_all = cached_bonds(open(local_bonds, "rb").read(), local_bonds)
    else:
        st.title("Green bond issuers: do their stocks outperform?")
        st.markdown(f"""
**To use your own data**

1. Download the ESG bond list from
   [Euronext](https://live.euronext.com/en/products/fixed-income/esg-bonds)
   (*Download ESG Bonds list*). Put it in `{DATA_DIR}` or upload it in the sidebar.
2. Find each issuer's stock ticker (sidebar button), then review the table in the *Sample* tab.
3. Give a list of control firms (non-issuers), e.g. `control_universe.csv`.
4. Download prices and factors, or put `prices.csv` and the factor files in `{DATA_DIR}`.
""")
        st.stop()
    issuers_all = eng.first_issues(eng.filter_bonds(bonds_all, settings))

    # 2. issuer -> ticker table ---------------------------------------------------------
    st.sidebar.markdown("**2. Issuer → ticker table**")
    map_up = st.sidebar.file_uploader("Reviewed ticker table (.csv)", ["csv"], key="map")
    if map_up is not None:
        st.session_state["mapping"] = pd.read_csv(map_up)
    elif "mapping" not in st.session_state and first_existing("issuer_tickers.csv"):
        st.session_state["mapping"] = pd.read_csv(first_existing("issuer_tickers.csv"))
    if st.sidebar.button(f"Search Yahoo for {len(issuers_all)} issuers"):
        bar = st.sidebar.progress(0.0)
        st.session_state["mapping"] = eng.suggest_tickers(
            issuers_all["issuer"].tolist(), cache_path=str(DATA_DIR / "ticker_search_cache.json"),
            progress=lambda i, n, x: bar.progress((i + 1) / n, text=x[:40]))
    mapping = st.session_state.get("mapping")
    if mapping is None:
        st.warning("Load a ticker table or search Yahoo for tickers (sidebar, step 2).")
        st.dataframe(issuers_all, width="stretch")
        st.stop()
    mapping["include"] = mapping["include"].astype(str).str.lower().isin(["true", "1", "yes"])

    # 3. control universe -------------------------------------------------------------------
    st.sidebar.markdown("**3. Control universe (non-issuers)**")
    uni_up = st.sidebar.file_uploader("Control tickers (.csv with a 'ticker' column)", ["csv"], key="uni")
    uni_file = uni_up if uni_up is not None else first_existing("control_universe.csv")
    default_uni = "\n".join(pd.read_csv(uni_file)["ticker"].astype(str)) if uni_file else ""
    uni_text = st.sidebar.text_area("Tickers, one per line or comma separated", default_uni, height=100)
    universe = parse_tickers(uni_text)

    # 4. prices and factors -------------------------------------------------------------------
    st.sidebar.markdown("**4. Prices and factors**")
    region = st.sidebar.selectbox("Factor region", ["Europe", "Developed"])
    fm_file = first_existing(f"{region}_5_Factors_CSV.zip", f"{region}_5_Factors.csv")
    fd_file = first_existing(f"{region}_5_Factors_Daily_CSV.zip", f"{region}_5_Factors_Daily.csv")
    try:
        factors_m = eng.read_french_file(fm_file) if fm_file else cached_factors(region, "monthly")
        factors_d = eng.read_french_file(fd_file) if fd_file else cached_factors(region, "daily")
    except Exception as e:
        st.sidebar.error(f"Factor download failed ({e}). Put the Kenneth French zip files in "
                         f"{DATA_DIR} (a VPN may be needed to download them).")
        st.stop()

    inc = mapping[mapping["include"] & mapping["ticker"].notna()]
    need = tuple(sorted(set(inc["ticker"]) | set(universe)))
    price_up = st.sidebar.file_uploader("Prices CSV (date + one column per ticker)", ["csv"], key="px")
    if price_up is not None:
        prices = eng.read_prices_csv(price_up)
    elif st.sidebar.button(f"Download prices for {len(need)} tickers"):
        prices = cached_prices(need, str(pd.Timestamp(start) - pd.DateOffset(years=4))[:10], str(end))
        prices.to_csv(DATA_DIR / "prices.csv")
    elif first_existing("prices.csv.gz", "prices.csv"):
        prices = cached_prices_file(first_existing("prices.csv.gz", "prices.csv"))
    if prices is None:
        st.warning("Download or upload prices (sidebar, step 4).")
        st.stop()
    missing = [t for t in need if t not in prices.columns]
    if missing:
        st.sidebar.caption(f"{len(missing)} tickers have no prices and are skipped.")

if not saved:
    bonds = eng.filter_bonds(bonds_all, settings)
    issuers = eng.first_issues(bonds)
    events = issuers.merge(mapping[mapping["include"].astype(bool)][["issuer", "ticker"]],
                           on="issuer", how="left")
    res = cached_analysis(events, prices, factors_m, factors_d, tuple(universe), settings.to_json())
regs, ports = res["regressions"], res["portfolios"]

# --- Dashboard header ---
st.title("Green bond issuers: do their stocks outperform?")
main = regs[(regs.portfolio == "green_minus_control") & (regs.model == model)]
main_green = regs[(regs.portfolio == "green") & (regs.model == model)]
k = st.columns(5)
k[0].metric("Issuers in sample", f"{len(issuers)}")
k[1].metric("With price data", f"{res['events']['ticker'].notna().sum()}")
k[2].metric("Matched controls", f"{res['matches']['control'].notna().sum()}")
if not main_green.empty and "alpha_ann_pct" in main_green:
    k[3].metric(f"Issuer portfolio {model} alpha", f"{main_green['alpha_ann_pct'].iloc[0]:.2f}% / yr",
                f"t = {main_green['t_alpha'].iloc[0]:.2f}", delta_color="off", delta_arrow="off")
if not main.empty and "alpha_ann_pct" in main:
    sig = main["p_alpha"].iloc[0] < alpha_level
    k[4].metric(f"Issuers minus controls ({model})", f"{main['alpha_ann_pct'].iloc[0]:.2f}% / yr",
                ("significant" if sig else "not significant") + f" at {alpha_level:.0%}",
                delta_color="off", delta_arrow="off")

tabs = st.tabs(["Portfolios", "Factor regressions", "Event study", "Robustness",
                "Matching", "Sample", "Method & export"])

# --- Portfolio results ---
with tabs[0]:
    cum = (1 + ports[["green", "control"]].fillna(0)).cumprod() - 1
    fig = go.Figure()
    fig.add_scatter(x=cum.index, y=cum["green"], name="Green bond issuers", line=dict(color=C_GREEN, width=2))
    if ports["control"].notna().any():
        fig.add_scatter(x=cum.index, y=cum["control"], name="Matched controls",
                        line=dict(color=C_CONTROL, width=2))
    st.subheader("Cumulative return")
    st.plotly_chart(style_fig(fig, y_pct=True), width="stretch")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=ports.index, y=ports["green_minus_control"], marker_color=C_DIFF,
                name="Issuers minus controls")
    c1.subheader("Monthly return difference")
    c1.plotly_chart(style_fig(fig, 300, y_pct=True), width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=ports.index, y=ports["n_green"], name="Issuers held", line=dict(color=C_GREEN, width=2))
    fig.add_scatter(x=ports.index, y=ports["n_control"], name="Controls held", line=dict(color=C_CONTROL, width=2))
    c2.subheader("Number of firms in each portfolio")
    c2.plotly_chart(style_fig(fig, 300), width="stretch")

    st.subheader("Performance")
    st.dataframe(res["performance"].round(2), width="stretch", hide_index=True)

# --- Factor regressions ---
with tabs[1]:
    st.subheader("Alpha by portfolio and factor model")
    show = regs.copy()
    show["significant"] = show["p_alpha"] < alpha_level
    cols = ["portfolio", "model", "n_months", "alpha_m_pct", "alpha_ann_pct", "t_alpha",
            "p_alpha", "significant", "r2"] + [c for c in show.columns if c.startswith("b_")]
    st.dataframe(show[[c for c in cols if c in show]].round(3), width="stretch", hide_index=True)

    fig = go.Figure()
    for name, col in (("green", C_GREEN), ("control", C_CONTROL), ("green_minus_control", C_DIFF)):
        d = regs[regs.portfolio == name]
        if d.empty or "alpha_ann_pct" not in d:
            continue
        se = d["alpha_ann_pct"] / d["t_alpha"]
        fig.add_bar(x=d["model"], y=d["alpha_ann_pct"], name=name, marker_color=col,
                    error_y=dict(type="data", array=1.96 * se.abs(), color=MUTED))
    fig.update_layout(barmode="group")
    st.subheader("Annualised alpha with 95% interval")
    st.plotly_chart(style_fig(fig, 360, "% per year"), width="stretch")
    st.caption("Newey-West standard errors. The long-short portfolio (issuers minus controls) "
               "is a zero-investment return, so it is not reduced by the risk-free rate.")

# --- Event study ---
with tabs[2]:
    es = res["event_study"]
    if es is None or es["per_event"].empty:
        st.info("No event study: daily factors are missing or no event has enough data.")
    else:
        c1, c2 = st.columns([3, 2])
        path = es["caar_path"]
        fig = go.Figure()
        fig.add_vrect(x0=windows[0][0] - 0.5, x1=windows[0][1] + 0.5, fillcolor=GRID, opacity=0.5,
                      line_width=0)
        fig.add_scatter(x=path.index, y=path.values, mode="lines+markers", name="CAAR",
                        line=dict(color=C_GREEN, width=2), marker=dict(size=8))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        c1.subheader("Average cumulative abnormal return around the listing day")
        c1.plotly_chart(style_fig(fig, 360, y_pct=True).update_xaxes(title="trading days from listing"),
                        width="stretch")
        c2.subheader("Summary")
        summ = es["summary"].copy()
        summ["significant"] = summ["t_stat"].abs() > {0.01: 2.576, 0.05: 1.96, 0.10: 1.645}[alpha_level]
        c2.dataframe(summ.round(3), width="stretch", hide_index=True)
        col = summ["window"].iloc[0]
        fig = go.Figure(go.Histogram(x=es["per_event"][col], marker_color=C_CONTROL, nbinsx=30))
        c2.plotly_chart(style_fig(fig, 220, f"events").update_xaxes(title=col, tickformat=".1%"),
                        width="stretch")
        with st.expander("Every event"):
            st.dataframe(es["per_event"].round({c: 4 for c in es["per_event"].select_dtypes("number").columns}), width="stretch", hide_index=True)

# --- Robustness checks ---
with tabs[3]:
    st.subheader(f"Issuers-minus-controls {model} alpha (% per year) across choices")
    grid = None
    if saved:
        grid = snap["grid"]
        if grid is not None and "model" in grid:
            grid = grid[grid["model"] == model]
        st.caption("Saved grid: every other setting as in the sidebar.")
    else:
        c1, c2 = st.columns(2)
        holds = c1.multiselect("Holding periods (months)", [6, 12, 24, 36, 48, 60, 84, 120], [12, 24, 36, 60])
        wins = c2.multiselect("Winsorising (%)", [0.0, 0.5, 1.0, 2.5, 5.0], [0.0, 1.0, 5.0])
        if universe and st.button("Run grid"):
            grid = cached_grid(events, prices, factors_m, tuple(universe), settings.to_json(),
                               tuple(sorted(holds)), tuple(sorted(wins)))
        elif not universe:
            st.info("Add a control universe to run the grid.")
    if grid is not None:
        if not grid.empty:
            z = grid.pivot(index="hold_months", columns="winsor_pct", values="alpha_ann_pct")
            t = grid.pivot(index="hold_months", columns="winsor_pct", values="t_alpha")
            text = z.round(2).astype(str) + "<br>t=" + t.round(2).astype(str)
            fig = go.Figure(go.Heatmap(z=z.values, x=[f"{c}%" for c in z.columns],
                                       y=[f"{i} m" for i in z.index], text=text.values,
                                       texttemplate="%{text}", colorscale="RdBu", zmid=0))
            fig.update_xaxes(type="category", title="winsorising (each tail)")
            fig.update_yaxes(type="category", title="holding period")
            st.plotly_chart(style_fig(fig, 360).update_layout(hovermode="closest"),
                            width="stretch")
            st.dataframe(grid.round(3), width="stretch", hide_index=True)

# --- Issuer matching ---
with tabs[4]:
    m = res["matches"].dropna(subset=["control"])
    st.subheader("Issuer and control firm, matched on pre-event beta and volatility")
    if m.empty:
        st.info("No matches. Add a control universe with enough price history.")
    else:
        c1, c2 = st.columns(2)
        for c, var in ((c1, "beta"), (c2, "vol")):
            fig = go.Figure()
            fig.add_scatter(x=m[f"{var}_issuer"], y=m[f"{var}_control"], mode="markers",
                            marker=dict(color=C_CONTROL, size=8), name=var,
                            text=m["ticker"] + " / " + m["control"])
            lo, hi = m[[f"{var}_issuer", f"{var}_control"]].min().min(), m[[f"{var}_issuer", f"{var}_control"]].max().max()
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dash"),
                            name="perfect match")
            c.markdown(f"**{var}: issuer vs control**")
            c.plotly_chart(style_fig(fig, 320).update_layout(hovermode="closest").update_xaxes(title="issuer"),
                           width="stretch")
        st.dataframe(m.round(3), width="stretch", hide_index=True)

# --- Sample overview ---
with tabs[5]:
    c1, c2, c3 = st.columns(3)
    if saved:
        c1.metric("Bonds in the file", meta.get("bonds_in_file", "?"))
        c2.metric("Bonds after filters", meta.get("bonds_after_filters", "?"))
        c3.metric("Public issuers' bonds excluded", meta.get("public_excluded", "?"))
    else:
        c1.metric("Bonds in the file", len(bonds_all))
        c2.metric("Bonds after filters", len(bonds))
        c3.metric("Public issuers' bonds excluded",
                  int(eng.filter_bonds(bonds_all, replace(settings, exclude_public_issuers=False))["is_public"].sum())
                  if excl_public else 0)
    per_year = issuers.assign(year=issuers["event_date"].dt.year).groupby("year").size()
    fig = go.Figure(go.Bar(x=per_year.index, y=per_year.values, marker_color=C_GREEN))
    st.subheader("New issuers per year (first bond in sample)")
    st.plotly_chart(style_fig(fig, 300, "issuers"), width="stretch")
    st.subheader("Issuer → ticker table")
    if mode == "My data":
        edited = st.data_editor(mapping, width="stretch", hide_index=True,
                                column_config={"include": st.column_config.CheckboxColumn()})
        st.session_state["mapping"] = edited
        st.download_button("Download ticker table", edited.to_csv(index=False), "issuer_tickers.csv",
                           mime="text/csv", key="dl_tickers", on_click="ignore")
    elif mapping is not None:
        st.dataframe(mapping, width="stretch", hide_index=True)
    if not saved:
        with st.expander("Bonds after filters"):
            st.dataframe(bonds, width="stretch", hide_index=True)

# --- Methodology and export ---
with tabs[6]:
    st.markdown(f"""
**Method**

* **Events.** An issuer enters the sample on the listing date of its first bond of the selected
  types between {settings.start} and {settings.end}.
* **Portfolios.** From {settings.entry_lag_months} month(s) after the listing month, each issuer is
  held for {settings.hold_months} months in an equal-weighted portfolio (calendar-time approach).
  Each issuer's matched control is held over the same months.
* **Matching.** The control is the non-issuer with the closest beta and volatility over the
  {settings.match_window_months} months before the listing (standardised Euclidean distance).
* **Risk adjustment.** Monthly excess returns are regressed on CAPM, Fama-French three- and
  five-factor models, with Newey-West ({settings.nw_lags} lags) standard errors.
* **Event study.** A market model is estimated over trading days {settings.est_window[0]} to
  {settings.est_window[1]}; abnormal returns are summed over each event window and tested with a
  cross-sectional t-test.

This follows the logic of Badía, Cortez and Silva (2024, *Economics Letters*): green bond issuers
compared with matched non-issuers on risk-adjusted returns. The matching rule and event study are
choices made for this tool, not taken from that paper.
""")
    def results_excel() -> bytes:
        """Build the Excel file only when the button is clicked."""
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            regs.to_excel(xw, sheet_name="regressions", index=False)
            ports.to_excel(xw, sheet_name="portfolios")
            res["performance"].to_excel(xw, sheet_name="performance", index=False)
            res["matches"].to_excel(xw, sheet_name="matches", index=False)
            if res["event_study"] is not None:
                res["event_study"]["summary"].to_excel(xw, sheet_name="event_summary", index=False)
                res["event_study"]["per_event"].to_excel(xw, sheet_name="events", index=False)
            pd.DataFrame({"setting": list(eng.asdict(settings).keys()),
                          "value": [str(v) for v in eng.asdict(settings).values()]}).to_excel(
                xw, sheet_name="settings", index=False)
        return buf.getvalue()

    # on_click="ignore": a click downloads the file without rerunning the app, so
    # changing a setting afterwards never starts another download
    st.download_button("Download all results (Excel)", results_excel, "green_bond_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="dl_results", on_click="ignore")
    st.download_button("Download settings (JSON)", settings.to_json(), "settings.json",
                       mime="application/json", key="dl_settings", on_click="ignore")

