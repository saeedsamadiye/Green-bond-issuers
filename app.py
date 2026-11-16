"""
app.py - dashboard for greenbond_engine.py

Run with:   streamlit run app.py

Three data modes:
  Saved results   the real-data results stored in data/saved_results.json
                  (works online, without the price file)
  Demo data       made-up stocks with known true effects (offline)
  My data         recomputes everything from the files in data/

Author: Saeed Samadiye
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import greenbond_engine as eng

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SNAPSHOT_FILE = DATA_DIR / "saved_results.json"

st.set_page_config(page_title="Green Bond Issuers", page_icon="🌿", layout="wide")

C_GREEN, C_CONTROL, C_DIFF = "#1baf7a", "#2a78d6", "#eb6834"
MUTED, GRID = "#6b6a64", "#e4e3de"
LEVEL_NAMES = {"sector+country": "same sector and country", "sector+region": "same sector and region",
               "sector": "same sector", "country": "same country", "region": "same region", "pool": "whole pool"}
MATCH_MODES = {"sector_country": "Same sector first (country, region, anywhere), then country, region, anyone",
               "country_region": "Same country, then same region, then anyone",
               "pool": "Nearest firm anywhere (version 1)"}


def first_existing(*names):
    for n in names:
        if (DATA_DIR / n).exists():
            return str(DATA_DIR / n)
    return None


def style_fig(fig: go.Figure, height: int = 380, y_title: str = "", y_pct: bool = False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED, title=y_title, tickformat=".1%" if y_pct else None)
    return fig


# --- cached loaders ---------------------------------------------------------------

@st.cache_data(show_spinner="Building demo data...")
def cached_demo(seed, alpha, announce, vol):
    return eng.make_demo_data(seed=seed, green_alpha_ann=alpha / 100, announce_effect=announce / 100,
                              idio_vol_daily=vol / 100)


@st.cache_data(show_spinner="Reading the data folder...")
def cached_folder(folder: str, region: str, stamp: float):
    return eng.load_data_folder(folder, region)


@st.cache_data(show_spinner="Loading saved results...")
def cached_snapshot(path: str, mtime: float):
    return eng.load_snapshot(path)


@st.cache_data(show_spinner="Running the analysis...")
def cached_analysis(events, prices, fm, fd, universe: tuple, settings_json: str, fx, info, earnings, ann):
    s = eng.settings_from_dict(json.loads(settings_json))
    return eng.run_analysis(events, prices, fm, fd, list(universe), s, fx=fx, info=info,
                            earnings=earnings, announcements=ann)


@st.cache_data(show_spinner="Running the robustness grid...")
def cached_grid(events, prices, fm, universe: tuple, settings_json: str, holds, winsors, fx, info):
    s = eng.settings_from_dict(json.loads(settings_json))
    return eng.robustness_grid(events, prices, fm, list(universe), s, holds, winsors, fx=fx, info=info)


@st.cache_data(show_spinner="Running the four designs...")
def cached_comparison(events, prices, fm, fd, universe: tuple, settings_json: str, fx, info, earnings, ann):
    s = eng.settings_from_dict(json.loads(settings_json))
    return eng.compare_designs(events, prices, fm, fd, list(universe), s, fx=fx, info=info,
                               earnings=earnings, announcements=ann)


def parse_windows(text: str) -> tuple:
    out = []
    for part in text.split(";"):
        a, b = part.strip().strip("[]()").split(",")
        out.append((int(a), int(b)))
    return tuple(out)


# --- sidebar ----------------------------------------------------------------------

sb = st.sidebar
sb.title("🌿 Green Bond Issuers")
MODES = (["Saved results (real data)"] if SNAPSHOT_FILE.exists() else []) + ["Demo data (offline)", "My data"]
mode = sb.radio("Data", MODES, help="Saved results: real-data results of the default run. "
                                    "Demo data: made-up stocks with known true effects. "
                                    "My data: recompute everything from the files in data/.")
saved = mode.startswith("Saved")
snap = cached_snapshot(str(SNAPSHOT_FILE), SNAPSHOT_FILE.stat().st_mtime) if saved else None
S0 = snap["settings"] if saved else eng.Settings()
lock = saved            # in Saved mode only display choices can change
if saved:
    sb.caption("Settings are locked to the saved run. Switch to *My data* to change them.")

sb.header("Sample")
c1, c2 = sb.columns(2)
start = c1.date_input("Start", pd.Timestamp(S0.start), disabled=lock)
end = c2.date_input("End", pd.Timestamp(S0.end), disabled=lock)
bond_types = sb.multiselect("Bond types counted as events", list(eng.BOND_TYPES), list(S0.bond_types),
                            disabled=lock)
min_nominal = sb.number_input("Minimum bond size (millions)", 0.0, 10000.0, float(S0.min_nominal_mm), 50.0,
                              disabled=lock)
excl_public = sb.checkbox("Exclude governments, agencies, supranationals", S0.exclude_public_issuers,
                          disabled=lock)

sb.header("Currency")
currency = sb.radio("Share returns in", ["USD", "local"], index=["USD", "local"].index(S0.currency),
                    format_func=lambda c: "US dollars (same as the factors)" if c == "USD"
                    else "each share's own currency (version 1)", disabled=lock)

sb.header("Portfolios")
hold = sb.slider("Holding period after first green bond (months)", 6, 120, S0.hold_months, 6, disabled=lock)
lag = sb.number_input("Start holding N months after the listing month", 0, 12, S0.entry_lag_months,
                      disabled=lock)
min_firms = sb.number_input("Drop months with fewer firms than", 1, 50, S0.min_firms, disabled=lock)
winsor = sb.select_slider("Winsorise monthly returns (each tail, %)", [0.0, 0.5, 1.0, 2.5, 5.0],
                          float(S0.winsor_pct), disabled=lock)

sb.header("Matching")
match_mode = sb.radio("Where to look for each issuer's twin", list(MATCH_MODES),
                      index=list(MATCH_MODES).index(S0.match_mode), disabled=lock,
                      format_func=MATCH_MODES.get)
caliper = sb.slider("Largest distance accepted in a country or region", 0.05, 1.0,
                    float(S0.match_caliper), 0.05, disabled=lock or match_mode == "pool",
                    help="Standardised distance in beta and volatility. If the best twin at one step "
                         "(e.g. same country) is further than this, the search moves to the next step.")
match_win = sb.slider("Pre-event window for beta and volatility (months)", 12, 60, S0.match_window_months, 6,
                      disabled=lock)
match_min = sb.number_input("Minimum months of data to match", 6, 60, S0.match_min_months, disabled=lock)
replacement = sb.checkbox("Allow one twin for several issuers", S0.match_with_replacement, disabled=lock)
pairs_only = sb.checkbox("Matched pairs only", S0.matched_pairs_only, disabled=lock,
                         help="Hold an issuer only in months when its twin is held too. Issuers without a twin "
                              "are left out, so both portfolios always contain the same number of firms.")

sb.header("Regressions")
model = sb.selectbox("Main factor model", list(eng.FACTOR_MODELS), index=list(eng.FACTOR_MODELS).index(S0.factor_model))
nw = sb.number_input("Newey-West lags", 0, 24, S0.nw_lags, disabled=lock)
alpha_level = sb.select_slider("Significance level for the tables", [0.01, 0.05, 0.10], 0.05,
                                help="Sets the true/false 'significant' column in the tables. The header shows all three levels.")

sb.header("Event study")
date_source = sb.radio("Event date", ["listing", "announcement"],
                       index=["listing", "announcement"].index(S0.event_date_source), disabled=lock,
                       format_func=lambda x: "Listing date (Euronext)" if x == "listing"
                       else "Announcement date where filled in (data/announcement_dates.csv)")
excl_earn = sb.checkbox("Drop events with an earnings release in the window", S0.exclude_earnings,
                        disabled=lock, help="Needs data/earnings_dates.csv (made by --update-data).")
est = sb.slider("Estimation window (trading days)", -400, -10, tuple(S0.est_window), disabled=lock)
win_text = sb.text_input("Event windows (a,b; c,d)", "; ".join(f"{a},{b}" for a, b in S0.event_windows),
                         disabled=lock)
try:
    windows = parse_windows(win_text)
except Exception:
    sb.error("Write windows like  -1,1; 0,5")
    windows = ((-1, 1),)

settings = replace(S0, start=str(start), end=str(end), bond_types=tuple(bond_types) or ("Green",),
                   min_nominal_mm=float(min_nominal), exclude_public_issuers=excl_public, currency=currency,
                   hold_months=int(hold), entry_lag_months=int(lag), min_firms=int(min_firms),
                   winsor_pct=float(winsor), match_mode=match_mode, match_caliper=float(caliper),
                   match_window_months=int(match_win), match_min_months=int(match_min),
                   match_with_replacement=replacement, matched_pairs_only=pairs_only, factor_model=model, nw_lags=int(nw),
                   event_date_source=date_source, exclude_earnings=excl_earn, est_window=tuple(est),
                   event_windows=windows)

# --- data ---------------------------------------------------------------------------

extra = {"fx": None, "info": None, "earnings": None, "announcements": None}
comparison = None

if saved:
    meta, res, issuers = snap["meta"], snap["res"], snap["issuers"]
    comparison = snap["comparison"]
    mapping = pd.read_csv(first_existing("issuer_tickers.csv")) if first_existing("issuer_tickers.csv") else None
    universe = ["saved"]
    st.info(f"**Saved results (real data).** Computed on {meta.get('computed_on', '?')} with the default "
            f"settings shown in the sidebar: Euronext ESG bond list, Yahoo Finance prices (to "
            f"{meta.get('prices_to', '?')}) converted to US dollars with ECB exchange rates (to "
            f"{meta.get('fx_to', '?')}), and Kenneth French {meta.get('factor_region', 'Europe')} factors "
            f"(to {meta.get('factors_to', '?')}). To change settings, run the app locally in *My data* mode.")
elif mode.startswith("Demo"):
    with sb.expander("Demo settings (true values)"):
        d_alpha = st.slider("True extra return of issuers (% per year)", -5.0, 5.0, 2.0, 0.5)
        d_ann = st.slider("True listing-day reaction (%)", -1.0, 1.0, 0.4, 0.1)
        d_vol = st.slider("Firm-specific daily volatility (%)", 0.2, 2.5, 1.2, 0.1,
                          help="Higher is noisier and closer to real shares.")
        d_seed = st.number_input("Random seed", 0, 999, 7)
    demo = cached_demo(int(d_seed), d_alpha, d_ann, d_vol)
    bonds_all = eng.load_bond_list(io.BytesIO(eng.bonds_to_excel_bytes(demo["bonds"])))
    mapping = demo["mapping"].assign(include=True)
    prices, factors_m, factors_d, universe = demo["prices"], demo["factors_m"], demo["factors_d"], demo["universe"]
    extra.update(fx=demo["fx"], info=demo["info"])
    st.info(f"**Demo data.** Made-up shares with a true extra return of {d_alpha:.1f}% a year after the first "
            f"green bond and a true listing-day reaction of {d_ann:.1f}%.")
else:
    region = sb.selectbox("Factor region", ["Europe", "Developed"])

    def run_update():
        """Download missing prices, exchange rates, firm info, earnings dates and factors."""
        import contextlib
        buf = io.StringIO()
        with st.spinner("Downloading data (this takes a few minutes)..."):
            with contextlib.redirect_stdout(buf):
                eng.update_data(str(DATA_DIR))
        cached_folder.clear()
        return "\n".join(l for l in buf.getvalue().splitlines() if l[:2] in ("1.", "2.", "3.", "4.", "5."))

    with sb.expander("Update data from the internet"):
        st.caption("Downloads prices for any missing ticker, ECB exchange rates, firm country and sector, "
                   "earnings dates and the latest French factors into the data folder.")
        if st.button("Download / update data", key="update_side"):
            st.session_state["update_log"] = run_update()
            st.rerun()
    if st.session_state.get("update_log"):
        sb.code(st.session_state["update_log"], language=None)

    BOND_FILES = ("Euronext-Green-Bond-List.xlsx", "esg_bonds.xlsx", "esg_bonds.csv")
    if first_existing(*BOND_FILES) is None or first_existing("prices.csv.gz", "prices.csv") is None:
        st.title("Green bond issuers: do their shares outperform?")
        st.warning("**My data** needs two files that are not in the public repository: the Euronext bond list "
                   "and the share prices. Add them here once; they stay in your `data` folder.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Euronext ESG bond list**")
            if first_existing(*BOND_FILES):
                st.success("Found.")
            else:
                st.markdown("Download the Excel file from the [Euronext ESG bonds page]"
                            "(https://live.euronext.com/en/products/fixed-income/esg-bonds) and upload it here.")
                up = st.file_uploader("Euronext bond list (.xlsx)", type=["xlsx", "csv"])
                if up is not None:
                    name = "Euronext-Green-Bond-List.xlsx" if up.name.lower().endswith(".xlsx") else "esg_bonds.csv"
                    (DATA_DIR / name).write_bytes(up.getvalue())
                    st.rerun()
        with c2:
            st.markdown("**2. Share prices and other downloads**")
            if first_existing("prices.csv.gz", "prices.csv"):
                st.success("Found.")
            else:
                st.markdown("Yahoo Finance prices may not be republished, so download them yourself "
                            "(free, a few minutes):")
                if st.button("Download prices and data", type="primary"):
                    st.session_state["update_log"] = run_update()
                    st.rerun()
        st.stop()

    stamp = max((p.stat().st_mtime for p in DATA_DIR.iterdir()), default=0)
    try:
        d = cached_folder(str(DATA_DIR), region, stamp)
    except FileNotFoundError as e:
        st.title("Green bond issuers: do their shares outperform?")
        st.error(f"A required file is missing: {e}")
        if region != "Europe":
            st.markdown(f"The {region} factor files are downloaded by **Update data from the internet** in the "
                        "sidebar.")
        st.stop()
    bonds_all, mapping, universe = d["bonds_all"], d["mapping"], d["universe"]
    prices, factors_m, factors_d = d["prices"], d["factors_m"], d["factors_d"]
    extra.update({k: d[k] for k in extra})
    if settings.currency == "USD" and extra["fx"] is None:
        st.error("Converting to US dollars needs data/ecb_fx_rates.csv. Use *Update data from the internet* "
                 "in the sidebar, or switch the currency to local.")
        st.stop()
    missing_px = [t for t in universe if t not in prices.columns]
    if missing_px:
        sb.caption(f"{len(missing_px)} pool firms have no prices yet; *Update data from the internet* "
                   "downloads them.")

if not saved:
    bonds = eng.filter_bonds(bonds_all, settings)
    issuers = eng.first_issues(bonds)
    events = issuers.merge(mapping[mapping["include"].astype(bool)][["issuer", "ticker"]], on="issuer", how="left")
    try:
        res = cached_analysis(events, prices, factors_m, factors_d, tuple(universe), settings.to_json(),
                              extra["fx"], extra["info"], extra["earnings"], extra["announcements"])
    except ValueError as e:
        st.title("Green bond issuers: do their shares outperform?")
        st.warning(f"**Nothing to analyse.** {e}")
        st.stop()
regs, ports = res["regressions"], res["portfolios"]

# --- header -------------------------------------------------------------------------

st.title("Green bond issuers: do their shares outperform?")
main = regs[(regs.portfolio == "green_minus_control") & (regs.model == model)]
main_green = regs[(regs.portfolio == "green") & (regs.model == model)]
def sig_card(col, label, alpha, t, p):
    """An alpha with a coloured significance badge (1%, 5%, 10% or not significant)."""
    if p < 0.01:
        text, bg, fg = "significant at 1%", "#0f6b43", "#ffffff"
    elif p < 0.05:
        text, bg, fg = "significant at 5%", "#1d8a5b", "#ffffff"
    elif p < 0.10:
        text, bg, fg = "significant at 10%", "#e0a326", "#ffffff"
    else:
        text, bg, fg = "not significant", "#d9d9d4", "#3a3a37"
    col.markdown(
        f"<div style='font-size:0.875rem;opacity:.75;margin-bottom:2px'>{label}</div>"
        f"<div style='font-size:2rem;line-height:1.2'>{alpha:.2f}% / yr</div>"
        f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:10px;"
        f"font-size:0.8rem;font-weight:600'>{text}</span>"
        f"<div style='font-size:0.78rem;opacity:.65;margin-top:3px'>t = {t:.2f}, p = {p:.2f}</div>",
        unsafe_allow_html=True)


k = st.columns(5)
k[0].metric("Issuers in sample", f"{len(issuers)}")
k[1].metric("With price data", f"{res['events']['ticker'].notna().sum()}")
k[2].metric("Matched twins", f"{res['matches']['control'].notna().sum()}")
for col, label, row in ((k[3], f"Issuer portfolio {model} alpha", main_green),
                        (k[4], f"Issuers minus twins ({model})", main)):
    if not row.empty and "alpha_ann_pct" in row and pd.notna(row["alpha_ann_pct"].iloc[0]):
        sig_card(col, label, row["alpha_ann_pct"].iloc[0], row["t_alpha"].iloc[0], row["p_alpha"].iloc[0])

tabs = st.tabs(["Portfolios", "Factor regressions", "Event study", "Robustness", "Matching",
                "Design comparison", "Sample", "Method & export"])

# --- portfolios ---------------------------------------------------------------------
with tabs[0]:
    cum = (1 + ports[["green", "control"]].fillna(0)).cumprod() - 1
    fig = go.Figure()
    fig.add_scatter(x=cum.index, y=cum["green"], name="Green bond issuers", line=dict(color=C_GREEN, width=2))
    if ports["control"].notna().any():
        fig.add_scatter(x=cum.index, y=cum["control"], name="Matched twins", line=dict(color=C_CONTROL, width=2))
    st.subheader(f"Cumulative return ({'US dollars' if settings.currency == 'USD' else 'local currencies'})")
    st.plotly_chart(style_fig(fig, y_pct=True), width="stretch")

    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(x=ports.index, y=ports["green_minus_control"], marker_color=C_DIFF,
                           name="Issuers minus twins"))
    c1.subheader("Monthly return difference")
    c1.plotly_chart(style_fig(fig, 300, y_pct=True), width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=ports.index, y=ports["n_green"], name="Issuers held", line=dict(color=C_GREEN, width=2))
    fig.add_scatter(x=ports.index, y=ports["n_control"], name="Twins held", line=dict(color=C_CONTROL, width=2))
    c2.subheader("Firms in each portfolio")
    c2.plotly_chart(style_fig(fig, 300), width="stretch")
    st.subheader("Performance")
    st.dataframe(res["performance"].round(2), width="stretch", hide_index=True)

# --- regressions ----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Alpha by portfolio and factor model")
    show = regs.copy()
    show["significant"] = show["p_alpha"] < alpha_level
    cols = ["portfolio", "model", "n_months", "alpha_m_pct", "alpha_ann_pct", "t_alpha", "p_alpha",
            "significant", "r2"] + [c for c in show.columns if c.startswith("b_")]
    st.dataframe(show[[c for c in cols if c in show]].round(3), width="stretch", hide_index=True)
    fig = go.Figure()
    for name, col in (("green", C_GREEN), ("control", C_CONTROL), ("green_minus_control", C_DIFF)):
        d_ = regs[regs.portfolio == name]
        if d_.empty or "alpha_ann_pct" not in d_:
            continue
        se = (d_["alpha_ann_pct"] / d_["t_alpha"]).abs()
        fig.add_bar(x=d_["model"], y=d_["alpha_ann_pct"], name=name, marker_color=col,
                    error_y=dict(type="data", array=1.96 * se, color=MUTED))
    fig.update_layout(barmode="group")
    st.subheader("Annualised alpha with 95% interval")
    st.plotly_chart(style_fig(fig, 360, "% per year"), width="stretch")
    st.caption("Newey-West standard errors. The long-short portfolio (issuers minus twins) costs nothing to "
               "set up, so the risk-free rate is not subtracted from it.")

# --- event study ------------------------------------------------------------------------
with tabs[2]:
    es = res["event_study"]
    if es is None or es["per_event"].empty:
        st.info("No event study: daily factors are missing or no event has enough data.")
    else:
        methods = [m for m in ("market", "twin") if m in set(es["summary"]["method"])]
        method = st.radio("Abnormal return", methods, horizontal=True,
                          format_func=lambda m: "Market model" if m == "market"
                          else "Twin-adjusted (issuer minus its twin, same day)")
        chk = es.get("earnings_check") or {}
        if chk.get("clear", 0) + chk.get("clash", 0) == 0:
            st.caption("Earnings filter: no earnings dates available, so no event was checked. "
                       "Run `py greenbond_engine.py --update-data` to download them.")
        else:
            st.caption(f"Earnings filter: {chk.get('clash', 0)} events had an earnings release in the "
                       f"window{' and were dropped' if settings.exclude_earnings else ''}; "
                       f"{chk.get('clear', 0)} were clear; {chk.get('unknown', 0)} could not be checked.")
        src = es["per_event"].drop_duplicates("ticker")["date_source"].value_counts()
        st.caption(f"Event dates: {src.get('listing', 0)} listing dates, {src.get('announcement', 0)} "
                   "announcement dates.")
        c1, c2 = st.columns([3, 2])
        path = es["caar_path"][method] if method in es["caar_path"] else pd.Series(dtype=float)
        fig = go.Figure()
        fig.add_vrect(x0=-1.5, x1=1.5, fillcolor=GRID, opacity=0.5, line_width=0)
        fig.add_scatter(x=path.index, y=path.values, mode="lines+markers", name="CAAR",
                        line=dict(color=C_GREEN if method == "market" else C_CONTROL, width=2))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        c1.subheader("Average cumulative abnormal return around the event day")
        c1.plotly_chart(style_fig(fig, 360, y_pct=True).update_xaxes(title="trading days from event"),
                        width="stretch")
        summ = es["summary"][es["summary"]["method"] == method].copy()
        crit = {0.01: 2.576, 0.05: 1.96, 0.10: 1.645}[alpha_level]
        summ["significant"] = summ["t_stat"].abs() > crit
        c2.subheader("Summary")
        c2.dataframe(summ.drop(columns="method").round(3), width="stretch", hide_index=True)
        pe = es["per_event"][es["per_event"]["method"] == method]
        col = "CAR[-1,1]" if "CAR[-1,1]" in pe else summ["window"].iloc[0]
        fig = go.Figure(go.Histogram(x=pe[col], marker_color=C_CONTROL, nbinsx=30))
        c2.plotly_chart(style_fig(fig, 220, "events").update_xaxes(title=col, tickformat=".1%"), width="stretch")
        with st.expander("Every event"):
            st.dataframe(pe.round(4), width="stretch", hide_index=True)

# --- robustness -------------------------------------------------------------------------
with tabs[3]:
    st.subheader(f"Issuers-minus-twins {model} alpha (% per year) across choices")
    grid = None
    if saved:
        grid = snap["grid"]
        if grid is not None and "model" in grid:
            grid = grid[grid["model"] == model]
    else:
        c1, c2 = st.columns(2)
        holds = c1.multiselect("Holding periods (months)", [6, 12, 24, 36, 48, 60, 84, 120], [12, 24, 36, 60])
        wins = c2.multiselect("Winsorising (%)", [0.0, 0.5, 1.0, 2.5, 5.0], [0.0, 1.0, 5.0])
        if st.button("Run grid"):
            grid = cached_grid(events, prices, factors_m, tuple(universe), settings.to_json(),
                               tuple(sorted(holds)), tuple(sorted(wins)), extra["fx"], extra["info"])
    if grid is not None and not grid.empty:
        z = grid.pivot(index="hold_months", columns="winsor_pct", values="alpha_ann_pct")
        t = grid.pivot(index="hold_months", columns="winsor_pct", values="t_alpha")
        text = z.round(2).astype(str) + "<br>t=" + t.round(2).astype(str)
        fig = go.Figure(go.Heatmap(z=z.values, x=[f"{c}%" for c in z.columns], y=[f"{i} m" for i in z.index],
                                   text=text.values, texttemplate="%{text}", colorscale="RdBu", zmid=0))
        fig.update_xaxes(type="category", title="winsorising (each tail)")
        fig.update_yaxes(type="category", title="holding period")
        st.plotly_chart(style_fig(fig, 360).update_layout(hovermode="closest"), width="stretch")
        st.dataframe(grid.round(3), width="stretch", hide_index=True)

# --- matching --------------------------------------------------------------------------
with tabs[4]:
    m = res["matches"].dropna(subset=["control"])
    if m.empty:
        st.info("No matches. Add a control pool with enough price history.")
    else:
        lv = m["match_level"].map(LEVEL_NAMES).value_counts()
        shown = [n for n in LEVEL_NAMES.values() if n in lv.index]
        c = st.columns(len(shown) + 2)
        for i, name in enumerate(shown):
            c[i].metric(f"Twins: {name}", int(lv.get(name, 0)))
        same_sector = (m["sector_issuer"].ne("") & m["sector_issuer"].eq(m["sector_control"])).sum() \
            if "sector_issuer" in m else 0
        c[-2].metric("Twin in the same sector", int(same_sector))
        c[-1].metric("Average distance", f"{m['distance'].mean():.2f}")
        c1, c2 = st.columns(2)
        for col_, var in ((c1, "beta"), (c2, "vol")):
            fig = go.Figure()
            fig.add_scatter(x=m[f"{var}_issuer"], y=m[f"{var}_control"], mode="markers",
                            marker=dict(color=C_CONTROL, size=8), name=var,
                            text=m["ticker"] + " / " + m["control"] + " (" + m["match_level"] + ")")
            lo = m[[f"{var}_issuer", f"{var}_control"]].min().min()
            hi = m[[f"{var}_issuer", f"{var}_control"]].max().max()
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dash"),
                            name="perfect match")
            col_.markdown(f"**{var}: issuer vs twin**")
            col_.plotly_chart(style_fig(fig, 320).update_layout(hovermode="closest").update_xaxes(title="issuer"),
                              width="stretch")
        st.subheader("Where the twins come from, by issuer country")
        st.dataframe(pd.crosstab(m["country_issuer"].replace("", "unknown"), m["match_level"].map(LEVEL_NAMES)),
                     width="stretch")
        st.dataframe(m.round(3), width="stretch", hide_index=True)

# --- design comparison ---------------------------------------------------------------------
with tabs[5]:
    st.subheader("What each fix changes")
    st.markdown("Same data, four designs: version 1 (local currencies, twin from the whole pool), "
                "dollars only, V2-C (dollars, twin from the same country or region first), and "
                "V2-S, the main design (dollars, twin from the same sector first).")
    if not saved and st.button("Run the four designs"):
        comparison = cached_comparison(events, prices, factors_m, factors_d, tuple(universe), settings.to_json(),
                                       extra["fx"], extra["info"], extra["earnings"], extra["announcements"])
    if comparison is not None:
        labels = {"issuer_alpha": "Issuer alpha (% / yr)", "issuer_t": "  t", "issuer_beta_mkt": "Issuer market beta",
                  "control_alpha": "Twin alpha (% / yr)", "control_t": "  t ",
                  "diff_alpha": "Issuers minus twins (% / yr)", "diff_t": "  t  ",
                  "twins_sector+country": "Twins: same sector and country",
                  "twins_sector+region": "Twins: same sector and region", "twins_sector": "Twins: same sector",
                  "twins_country": "Twins: same country", "twins_region": "Twins: same region",
                  "twins_pool": "Twins: whole pool", "twins_same_sector": "Twins in issuer's sector (any step)", "car_m1p1_market": "CAR[-1,1] market model (%)",
                  "car_m1p1_market_t": "  t   ", "car_m1p1_twin": "CAR[-1,1] twin-adjusted (%)",
                  "car_m1p1_twin_t": "  t    "}
        tab = comparison.set_index("design")[[c for c in labels if c in comparison]].rename(columns=labels).T
        st.dataframe(tab.round(2), width="stretch")
        st.caption(f"{settings.factor_model} model. The version 1 column uses version 1's method on today's "
                   "data folder, so it can differ slightly from the numbers first published.")

# --- sample -------------------------------------------------------------------------------
with tabs[6]:
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
    per_year = issuers.assign(year=pd.to_datetime(issuers["event_date"]).dt.year).groupby("year").size()
    fig = go.Figure(go.Bar(x=per_year.index, y=per_year.values, marker_color=C_GREEN))
    st.subheader("New issuers per year (first bond in sample)")
    st.plotly_chart(style_fig(fig, 300, "issuers"), width="stretch")
    if mapping is not None:
        st.subheader("Issuer → ticker table")
        st.dataframe(mapping, width="stretch", hide_index=True)

# --- method and export -----------------------------------------------------------------------
with tabs[7]:
    match_text = {
        "country_region": f"The twin comes from the same country if one is within distance {settings.match_caliper}, "
                          "otherwise from the same region, otherwise from the whole pool.",
        "sector_country": f"The twin comes from the same sector (and country, then region, then anywhere) if one is "
                          f"within distance {settings.match_caliper}, otherwise from the same country, region or "
                          "whole pool.",
        "pool": "The twin can come from anywhere in the pool.",
    }[settings.match_mode]
    st.markdown(f"""
**Method**

* **Events.** An issuer enters the sample on the listing date of its first bond of the selected types between
  {settings.start} and {settings.end} (or its announcement date, where one is filled in and chosen).
* **Currency.** {"Every share price is converted to US dollars with ECB reference rates before returns are computed, because the Kenneth French factors and risk-free rate are in US dollars." if settings.currency == "USD" else "Each share stays in its own currency (version 1). The factors are in US dollars, so this mixes currencies."}
* **Twins.** Each issuer gets one firm that never issued a green bond in the sample, with the closest market beta and
  volatility over the {settings.match_window_months} months before the event.
  {match_text}
* **Portfolios.** From {settings.entry_lag_months} month(s) after the event month, each issuer is held for
  {settings.hold_months} months in an equal-weighted portfolio; its twin is held over the same months.
  {"Only matched pairs are held: an issuer counts in a month only when its twin also has a return that month, so both portfolios hold the same number of firms." if settings.matched_pairs_only else "Issuers without a twin stay in the issuer portfolio, so it can hold a few more firms than the twin portfolio."}
* **Risk adjustment.** Monthly excess returns on CAPM, FF3 and FF5, with Newey-West ({settings.nw_lags} lags) errors.
* **Event study.** Market model estimated on trading days {settings.est_window[0]} to {settings.est_window[1]},
  and twin-adjusted returns (issuer minus twin on the same day). Windows: {", ".join(f"[{a}, {b}]" for a, b in settings.event_windows)}.

The design follows Badía, Cortez and Silva (2024, *Economics Letters*). The matching rule and the event study are
choices made for this project, not taken from that paper.
""")

    def results_excel() -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            regs.to_excel(xw, sheet_name="regressions", index=False)
            ports.to_excel(xw, sheet_name="portfolios")
            res["performance"].to_excel(xw, sheet_name="performance", index=False)
            res["matches"].to_excel(xw, sheet_name="matches", index=False)
            if res["event_study"] is not None:
                res["event_study"]["summary"].to_excel(xw, sheet_name="event_summary", index=False)
                res["event_study"]["per_event"].to_excel(xw, sheet_name="events", index=False)
            if comparison is not None:
                comparison.to_excel(xw, sheet_name="designs", index=False)
            pd.DataFrame({"setting": list(asdict(settings)), "value": [str(v) for v in asdict(settings).values()]}
                         ).to_excel(xw, sheet_name="settings", index=False)
        return buf.getvalue()

    st.download_button("Download all results (Excel)", results_excel, "green_bond_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="dl_results", on_click="ignore")
    st.download_button("Download settings (JSON)", settings.to_json(), "settings.json",
                       mime="application/json", key="dl_settings", on_click="ignore")
