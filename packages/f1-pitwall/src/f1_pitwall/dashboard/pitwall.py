"""Pit Wall v1 (spec §1.5): pick event + strategies, compare via Monte Carlo.

Run from the repo root:

    uv run streamlit run packages/f1-pitwall/src/f1_pitwall/dashboard/pitwall.py

The quantile model + CQR calibrator are trained once and cached
(``st.cache_resource``); moving a pit-window slider only re-runs the MC engine.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1_core.config import load_config
from f1_core.evaluate.calibration import BENCHMARK, _params_from_config, add_event_benchmark
from f1_core.models.cqr import CQRCalibrator, fit_cqr
from f1_core.models.quantile import TARGET, QuantileLapModel
from f1_core.models.splits import split_by_event
from f1_core.simulate.engine import RaceContext, Stint, simulate_race

CONFIG = load_config("configs/f1_core.yaml")
COMPOUNDS = CONFIG.ingest.dry_compounds


@st.cache_data
def load_clean_laps() -> pd.DataFrame:
    laps = pd.read_parquet(CONFIG.ingest.output_dir / "laps.parquet")
    return add_event_benchmark(laps[laps["is_clean"] & laps[TARGET].notna()].reset_index(drop=True))


@st.cache_resource
def trained_model() -> tuple[QuantileLapModel, CQRCalibrator]:
    clean = load_clean_laps()
    split = split_by_event(
        clean, calibration_fraction=0.2, test_fraction=0.0, seed=CONFIG.evaluation.split_seed
    )

    def norm(f: pd.DataFrame) -> pd.DataFrame:
        g = f.copy()
        g[TARGET] = g[TARGET] - g[BENCHMARK]
        return g

    model = QuantileLapModel(_params_from_config(CONFIG.model)).fit(norm(split.train))
    calibrator = fit_cqr(
        norm(split.calibration)[TARGET],
        model.predict(norm(split.calibration)),
        CONFIG.evaluation.target_coverage,
    )
    return model, calibrator


def strategy_controls(name: str, total_laps: int) -> tuple[Stint, ...]:
    st.subheader(f"Strategy {name}")
    c1, c2 = st.columns(2)
    first = c1.selectbox("Stint 1 tire", COMPOUNDS, index=1, key=f"{name}c1")
    second = c2.selectbox("Stint 2 tire", COMPOUNDS, index=2, key=f"{name}c2")
    stop = st.slider("Pit window (stop lap)", 5, total_laps - 5, total_laps // 2, key=f"{name}stop")
    return (Stint(str(first), stop), Stint(str(second), total_laps))


def main() -> None:
    st.set_page_config(page_title="Pit Wall", layout="wide")
    st.title("Pit Wall — strategy comparison (v1)")

    try:
        clean = load_clean_laps()
    except FileNotFoundError:
        st.error("No processed laps found — run `uv run python -m f1_core.ingest` first.")
        st.stop()

    events = sorted(clean["event"].unique())
    event = st.sidebar.selectbox("Event", events)
    laps_e = clean[clean["event"] == event]
    driver = st.sidebar.selectbox("Driver", sorted(laps_e["driver"].unique()))
    # Race length from the raw lap numbers seen in this event (clean subset is fine).
    total_laps = int(laps_e["lap_number"].max())

    ctx = RaceContext(
        event=event,
        season=int(laps_e["season"].iloc[0]),
        driver=driver,
        team=str(laps_e.loc[laps_e["driver"] == driver, "team"].mode().iloc[0]),
        track_temp=float(laps_e["track_temp"].median()),
        air_temp=float(laps_e["air_temp"].median()),
        total_laps=total_laps,
        benchmark_s=float(laps_e[BENCHMARK].iloc[0]),
    )

    left, right = st.columns(2)
    with left:
        strat_a = strategy_controls("A", total_laps)
    with right:
        strat_b = strategy_controls("B", total_laps)

    model, calibrator = trained_model()
    result = simulate_race(
        {"A": strat_a, "B": strat_b}, model, ctx, CONFIG.simulation, calibrator=calibrator
    )

    ca, cb = st.columns(2)
    ca.metric("P(A wins)", f"{result.win_probability('A'):.1%}")
    cb.metric("P(B wins)", f"{result.win_probability('B'):.1%}")

    fig = go.Figure()
    for name in ("A", "B"):
        fig.add_trace(
            go.Violin(y=result.total_time_distribution(name), name=name, box_visible=True)
        )
    fig.update_layout(yaxis_title="Total race time (s)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result.percentile_report().round(1))
    st.caption(
        f"{CONFIG.simulation.n_sims} sims · model cached, MC re-runs on every change · "
        "absolute times carry the static-benchmark offset (see artifacts/phase1/FINDINGS.md); "
        "A-vs-B comparison is unaffected."
    )


main()
