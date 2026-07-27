"""Live pit-wall dashboard (spec §3.3). Launched via ``pitwall --race ...``.

Panels: replay state (lap, positions), focal driver's current 5×5 energy frame,
strategy window band (p25–p75 of optimal stop) evolving lap by lap, alert feed.
Raw persistence outputs (delta, p_cliff) are shown next to the fused window so
miscalibration stays visible (spec §3.2).

Documented deviation: no live Grad-CAM overlay — that would import torch into a
process that already holds LightGBM (dual-libomp deadlock, see
artifacts/phase2/FINDINGS.md), and §2.5/2.6 showed the CNN is weak anyway.
"""

from __future__ import annotations

import argparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1_core.config import load_config
from f1_core.evaluate.calibration import add_event_benchmark
from f1_core.models.quantile import TARGET
from f1_pitwall.cli import resolve_event
from f1_pitwall.orchestrator.replay import TelemetryFrame, replay_race
from f1_pitwall.orchestrator.service import (
    PitwallOrchestrator,
    build_race_context,
    train_strategy_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--race", default="Bahrain 2024")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--driver", default="VER")
parser.add_argument("--config", default="configs/f1_core.yaml")
ARGS, _ = parser.parse_known_args()

CONFIG = load_config(ARGS.config)


@st.cache_data
def load_data(race: str) -> tuple[pd.DataFrame, pd.DataFrame, str, int, pd.DataFrame | None]:
    laps = pd.read_parquet(CONFIG.ingest.output_dir / "laps.parquet")
    event, season = resolve_event(laps, race)
    clean = add_event_benchmark(
        laps[laps["is_clean"] & laps[TARGET].notna()].reset_index(drop=True)
    )
    from f1_vision.train.dataset import sectors_path  # no torch in that module

    path = sectors_path(CONFIG.ingest.output_dir, season, event)
    sectors = pd.read_parquet(path) if path.exists() else None
    race_laps = laps[(laps["event"] == event) & (laps["season"] == season)]
    return race_laps, clean, event, season, sectors


@st.cache_resource
def anchor_model():
    laps = pd.read_parquet(CONFIG.ingest.output_dir / "laps.parquet")
    clean = add_event_benchmark(
        laps[laps["is_clean"] & laps[TARGET].notna()].reset_index(drop=True)
    )
    return train_strategy_model(clean, CONFIG)


def frame_figure(sectors: pd.DataFrame) -> go.Figure:
    """Current lap's 5×5 mean energy frame from its raw sector table."""
    from f1_vision.imaging.tab2img import CONTEXT_CHANNELS, lap_frames
    from f1_vision.proxies.physics import PROXY_CHANNELS

    table = sectors[PROXY_CHANNELS].copy()
    table = (table - table.min()) / (table.max() - table.min()).replace(0, 1)
    frames = lap_frames(table, dict.fromkeys(CONTEXT_CHANNELS, 0.5))
    fig = go.Figure(go.Heatmap(z=frames.mean(axis=0)[::-1], colorscale="Inferno", showscale=False))
    fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def main() -> None:
    st.set_page_config(page_title="Pit Wall Live", layout="wide")
    event_laps, clean, event, season, sectors = load_data(ARGS.race)
    drivers = sorted(event_laps["driver"].unique())
    focal = st.sidebar.selectbox(
        "Focal driver", drivers, index=drivers.index(ARGS.driver) if ARGS.driver in drivers else 0
    )
    fusion_on = st.sidebar.toggle("Fusion enabled", value=True)
    autoplay = st.sidebar.toggle("Auto-play", value=False)
    st.title(f"Pit Wall Live — {event}")

    key = f"{event}-{focal}-{fusion_on}"
    if st.session_state.get("key") != key:
        model, calibrator = anchor_model()
        ctx = build_race_context(clean, event, season, focal)
        st.session_state.update(
            key=key,
            events=replay_race(event_laps, sectors=sectors, speed=None),
            orch=PitwallOrchestrator(model, calibrator, ctx, CONFIG, fusion_enabled=fusion_on),
            history=[],
            alerts=[],
            frame=None,
            lap=0,
            done=False,
        )

    orch: PitwallOrchestrator = st.session_state.orch

    def advance_one_lap() -> None:
        for ev in st.session_state.events:
            if isinstance(ev, TelemetryFrame):
                if ev.driver == focal:
                    st.session_state.frame = ev.sectors
                continue
            orch.on_lap(ev)
            if ev.driver == focal:
                st.session_state.lap = ev.lap_number
                if ev.compound is not None:
                    update = orch.stop_window(focal, ev.lap_number, ev.compound)
                    if update is not None:
                        st.session_state.history.append(update)
                        if update.p_cliff > 0.6:
                            st.session_state.alerts.append(
                                f"lap {ev.lap_number}: cliff risk P={update.p_cliff:.2f} "
                                f"for {focal} — window p50 now lap {update.stop_p50:.0f}"
                            )
                return
        st.session_state.done = True

    cols = st.columns([1, 1, 5])
    if cols[0].button("Advance 1 lap", disabled=st.session_state.done):
        advance_one_lap()
    if cols[1].button("Advance 5", disabled=st.session_state.done):
        for _ in range(5):
            advance_one_lap()

    left, mid, right = st.columns([2, 2, 3])
    state = orch.state(focal)
    left.metric("Lap", st.session_state.lap)
    left.metric("Current pace delta (raw)", f"{state.current_delta:+.2f} s")
    left.metric("p_cliff (raw)", f"{state.p_cliff:.2f}")

    positions = pd.DataFrame(
        sorted(orch.positions.items(), key=lambda kv: kv[1]), columns=["driver", "position"]
    )
    mid.dataframe(positions, height=260, hide_index=True)

    if st.session_state.frame is not None:
        right.caption(
            "Current 5×5 energy frame (sector mean) — Grad-CAM overlay deferred, see FINDINGS"
        )
        right.plotly_chart(frame_figure(st.session_state.frame), use_container_width=True)

    history = st.session_state.history
    if history:
        h = pd.DataFrame([u.__dict__ for u in history])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.lap_number, y=h.stop_p75, line={"width": 0}, showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=h.lap_number,
                y=h.stop_p25,
                fill="tonexty",
                name="p25–p75 window",
                line={"width": 0},
                fillcolor="rgba(68,119,170,0.35)",
            )
        )
        fig.add_trace(go.Scatter(x=h.lap_number, y=h.stop_p50, name="p50 optimal stop"))
        fig.update_layout(
            height=330,
            xaxis_title="Race lap",
            yaxis_title="Recommended stop lap",
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.alerts:
        st.subheader("Alerts")
        for alert in reversed(st.session_state.alerts[-8:]):
            st.warning(alert)

    if autoplay and not st.session_state.done:
        advance_one_lap()
        st.rerun()


main()
